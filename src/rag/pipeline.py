"""
End-to-end RAG pipeline integration.

This module orchestrates the complete RAG pipeline from PDF processing
to answer generation, providing a high-level interface for the system.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.config import get_settings
from src.models import (
    Document,
    GeneratedAnswer,
    PDFChunk,
    PDFMetadata,
    ProcessingJob,
    ProcessingStatus,
)
from src.utils.errors import (
    PDFProcessingError,
    RAGPipelineError,
    VectorDatabaseError,
)
from src.utils.logging import LogContext, get_logger, log_performance
from src.google_drive.client import GoogleDriveClient
from src.google_drive.sync import DriveSyncManager
from src.pdf_processor.chunker import SemanticChunker
from src.pdf_processor.extractor import PDFExtractor
from src.pdf_processor.preprocessor import TextPreprocessor
from src.rag.embedder import EmbeddingGenerator
from src.rag.generator import AnswerGenerator
from src.rag.retriever import DocumentRetriever
from src.rag.hybrid_retriever import HybridGraphRetriever
from src.rag.hyde_enhancer import HyDERetrievalWrapper, HyDEEnhancer
from src.vector_db.base import VectorDatabase, VectorDatabaseFactory
from src.graph_db.base import GraphDatabase, GraphDatabaseFactory
from src.knowledge_graph.extractor import KnowledgeExtractor
from src.knowledge_graph.graphiti_builder import GraphitiBuilder

logger = get_logger(__name__)


class RAGPipeline:
    """
    Orchestrates the complete RAG pipeline.
    
    This class provides high-level methods for:
    - Processing PDFs from Google Drive or local files
    - Building and updating the knowledge base
    - Querying the system for answers
    - Managing the pipeline state
    """

    def __init__(
        self,
        vector_db: Optional[VectorDatabase] = None,
        graph_db: Optional[GraphDatabase] = None,
        pdf_extractor: Optional[PDFExtractor] = None,
        text_preprocessor: Optional[TextPreprocessor] = None,
        chunker: Optional[SemanticChunker] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        retriever: Optional[DocumentRetriever] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        drive_client: Optional[GoogleDriveClient] = None,
        sync_manager: Optional[DriveSyncManager] = None,
        knowledge_extractor: Optional[KnowledgeExtractor] = None,
        graphiti_builder: Optional[GraphitiBuilder] = None,
        use_hybrid_retrieval: bool = True,
        enable_hyde: bool = False,
        hyde_enhancer: Optional[HyDEEnhancer] = None,
    ) -> None:
        """
        Initialize RAG pipeline with components.
        
        All components are optional and will be created with defaults if not provided.
        """
        self.settings = get_settings()
        
        # Initialize components
        self.vector_db = vector_db
        self.graph_db = graph_db
        self.pdf_extractor = pdf_extractor or PDFExtractor()
        self.text_preprocessor = text_preprocessor or TextPreprocessor()
        self.chunker = chunker or SemanticChunker()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.use_hybrid_retrieval = use_hybrid_retrieval
        
        # Set up base retriever (hybrid or standard)
        base_retriever = None
        if use_hybrid_retrieval and self.settings.enable_knowledge_graph:
            base_retriever = retriever or HybridGraphRetriever(
                vector_db=self.vector_db,
                graph_db=self.graph_db,
                embedding_generator=self.embedding_generator,
            )
        else:
            base_retriever = retriever or DocumentRetriever()
        
        # Wrap with HyDE enhancement if enabled
        self.enable_hyde = enable_hyde
        self.hyde_enhancer = hyde_enhancer
        if enable_hyde:
            self.retriever = HyDERetrievalWrapper(
                base_retriever=base_retriever,
                hyde_enhancer=hyde_enhancer or HyDEEnhancer(),
                enable_hyde=True,
            )
        else:
            self.retriever = base_retriever
        
        self.answer_generator = answer_generator or AnswerGenerator()
        self.drive_client = drive_client
        self.sync_manager = sync_manager
        
        # Knowledge graph components
        self.knowledge_extractor = knowledge_extractor
        self.graphiti_builder = graphiti_builder
        
        # Processing state
        self._initialized = False
        self._processing_jobs: Dict[str, ProcessingJob] = {}

    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        if self._initialized:
            return
        
        logger.info("Initializing RAG pipeline")
        
        try:
            # Initialize vector database
            if not self.vector_db:
                self.vector_db = VectorDatabaseFactory.create(
                    self.settings.vector_db_type,
                    embedding_function=self.embedding_generator.generate_embeddings,
                )
            
            await self.vector_db.initialize()
            
            # Initialize graph database if enabled
            if self.settings.enable_knowledge_graph and not self.graph_db:
                self.graph_db = GraphDatabaseFactory.create("neo4j")
                await self.graph_db.initialize()
                
                # Initialize knowledge extraction components
                if not self.knowledge_extractor:
                    self.knowledge_extractor = KnowledgeExtractor()
                
                if not self.graphiti_builder:
                    self.graphiti_builder = GraphitiBuilder(
                        graph_db=self.graph_db,
                        embedding_generator=self.embedding_generator,
                    )
                    await self.graphiti_builder.initialize()
            
            # Initialize retriever with databases
            self.retriever.vector_db = self.vector_db
            self.retriever.embedding_generator = self.embedding_generator
            if hasattr(self.retriever, 'graph_db'):
                self.retriever.graph_db = self.graph_db
            await self.retriever.initialize()
            
            # Initialize Google Drive if configured
            if self.settings.drive_folder_ids and not self.drive_client:
                from src.google_drive.client import create_drive_client
                from src.google_drive.sync import create_sync_manager
                
                self.drive_client = create_drive_client()
                await self.drive_client.connect()
                
                self.sync_manager = create_sync_manager()
                self.sync_manager.drive_client = self.drive_client
            
            self._initialized = True
            logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise RAGPipelineError(f"Pipeline initialization failed: {str(e)}")

    @log_performance
    async def process_pdf_from_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
    ) -> Tuple[PDFMetadata, int]:
        """
        Process a PDF file and add to knowledge base.

        Args:
            file_path: Path to PDF file
            metadata: Additional metadata
            job_id: Optional job ID for tracking

        Returns:
            Tuple of (PDF metadata, number of chunks created)

        Raises:
            PDFProcessingError: If processing fails
        """
        await self.initialize()
        
        file_path = Path(file_path)
        job_id = job_id or f"file_{file_path.stem}"
        
        # Create processing job
        job = ProcessingJob(
            job_id=job_id,
            job_type="pdf_processing",
            status=ProcessingStatus.PROCESSING,
            metadata={"file_path": str(file_path), **(metadata or {})},
        )
        self._processing_jobs[job_id] = job
        
        try:
            with LogContext(job_id=job_id, file=file_path.name):
                # Extract PDF content
                pdf_metadata, pages = await self.pdf_extractor.extract_from_file(
                    file_path=file_path,
                    file_id=job_id,
                    drive_path=str(file_path),
                )
                
                # Process and store chunks
                chunks_created = await self._process_pdf_pages(
                    pdf_metadata,
                    pages,
                    metadata,
                    job,
                )
                
                # Update job status
                job.status = ProcessingStatus.COMPLETED
                job.progress = 100.0
                job.result = {
                    "pdf_id": pdf_metadata.file_id,
                    "chunks_created": chunks_created,
                    "pages_processed": len(pages),
                }
                
                logger.info(f"Successfully processed PDF: {file_path.name}")
                return pdf_metadata, chunks_created
                
        except Exception as e:
            job.status = ProcessingStatus.FAILED
            job.error = str(e)
            logger.error(f"Failed to process PDF: {e}")
            raise PDFProcessingError(f"Failed to process PDF: {str(e)}")

    @log_performance
    async def process_pdf_from_drive(
        self,
        file_id: str,
        job_id: Optional[str] = None,
    ) -> Tuple[PDFMetadata, int]:
        """
        Process a PDF from Google Drive.

        Args:
            file_id: Google Drive file ID
            job_id: Optional job ID

        Returns:
            Tuple of (PDF metadata, number of chunks created)
        """
        await self.initialize()
        
        if not self.drive_client:
            raise RAGPipelineError("Google Drive not configured")
        
        job_id = job_id or f"drive_{file_id}"
        
        # Create processing job
        job = ProcessingJob(
            job_id=job_id,
            job_type="drive_pdf_processing",
            status=ProcessingStatus.PROCESSING,
            metadata={"file_id": file_id},
        )
        self._processing_jobs[job_id] = job
        
        try:
            # Get file metadata
            file_metadata = await self.drive_client.get_file_metadata(file_id)
            
            with LogContext(job_id=job_id, file=file_metadata["name"]):
                # Download to stream
                pdf_stream, _ = await self.drive_client.download_pdf_stream(
                    file_id,
                    progress_callback=lambda p: setattr(job, "progress", p * 0.3),
                )
                
                # Extract content
                pdf_metadata, pages = await self.pdf_extractor.extract_from_stream(
                    pdf_stream=pdf_stream,
                    file_id=file_id,
                    filename=file_metadata["name"],
                    drive_path=f"drive://{file_id}",
                    file_size=int(file_metadata.get("size", 0)),
                )
                
                # Process chunks
                chunks_created = await self._process_pdf_pages(
                    pdf_metadata,
                    pages,
                    {"drive_metadata": file_metadata},
                    job,
                )
                
                # Mark as processed in Drive
                await self.drive_client.mark_as_processed(
                    file_id,
                    ProcessingStatus.COMPLETED,
                )
                
                # Update sync manager
                if self.sync_manager:
                    self.sync_manager.mark_file_processed(
                        file_id,
                        pdf_metadata,
                        chunks_created,
                    )
                
                job.status = ProcessingStatus.COMPLETED
                job.progress = 100.0
                
                return pdf_metadata, chunks_created
                
        except Exception as e:
            job.status = ProcessingStatus.FAILED
            job.error = str(e)
            
            # Mark as failed in Drive
            if self.drive_client:
                await self.drive_client.mark_as_processed(
                    file_id,
                    ProcessingStatus.FAILED,
                    str(e),
                )
            
            raise

    async def _process_pdf_pages(
        self,
        pdf_metadata: PDFMetadata,
        pages: List[Any],
        metadata: Optional[Dict[str, Any]],
        job: ProcessingJob,
    ) -> int:
        """Process PDF pages into chunks and store in vector database."""
        # Preprocess text
        processed_pages = []
        for page in pages:
            processed_text = self.text_preprocessor.preprocess(
                page.text,
                page.page_number,
            )
            processed_pages.append((page.page_number, processed_text))
        
        # Update progress
        job.progress = 30.0
        
        # Chunk the text
        all_chunks = await self.chunker.chunk_pages(
            processed_pages,
            pdf_id=pdf_metadata.file_id,
            metadata={
                "filename": pdf_metadata.filename,
                "total_pages": pdf_metadata.total_pages,
                **(metadata or {}),
            },
        )
        
        # Update progress
        job.progress = 40.0
        
        # Extract knowledge graph if enabled
        all_entities = []
        all_relationships = []
        
        if self.settings.enable_knowledge_graph and self.knowledge_extractor:
            logger.info("Extracting entities and relationships for knowledge graph")
            
            try:
                # Extract entities and relationships from chunks
                entities, relationships = await self.knowledge_extractor.extract_from_chunks(
                    all_chunks,
                    batch_size=5,
                    use_context=True,
                )
                
                all_entities = entities
                all_relationships = relationships
                
                logger.info(
                    f"Extracted {len(entities)} entities and {len(relationships)} relationships"
                )
                
                # Build knowledge graph if builder available
                if self.graphiti_builder and self.graph_db:
                    await self.graphiti_builder.build_from_chunks(
                        chunks=all_chunks,
                        pdf_metadata=pdf_metadata,
                        entities=entities,
                        relationships=relationships,
                    )
                else:
                    # Store in graph database directly
                    if self.graph_db:
                        # Bulk create entities
                        entity_ids = await self.graph_db.bulk_create_entities(entities)
                        
                        # Bulk create relationships
                        await self.graph_db.bulk_create_relationships(relationships)
                
                job.progress = 50.0
                
            except Exception as e:
                logger.error(f"Knowledge graph extraction failed: {e}")
                # Continue with vector storage even if graph fails
        
        # Convert chunks to documents
        documents = []
        for chunk in all_chunks:
            doc = Document(
                id=chunk.id,
                content=chunk.content,
                metadata={
                    "pdf_id": chunk.pdf_id,
                    "filename": pdf_metadata.filename,
                    "page_numbers": chunk.page_numbers,
                    "chunk_index": chunk.chunk_index,
                    "word_count": chunk.word_count,
                    "char_count": chunk.char_count,
                    "section_title": chunk.section_title,
                    "chapter_title": chunk.chapter_title,
                    **chunk.metadata,
                },
            )
            documents.append(doc)
        
        # Generate embeddings and store
        if documents:
            # Batch process for efficiency
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Generate embeddings
                texts = [doc.content for doc in batch]
                embeddings = await self.embedding_generator.generate_embeddings(texts)
                
                # Assign embeddings
                for doc, embedding in zip(batch, embeddings):
                    doc.embedding = embedding
                
                # Store in vector database
                await self.vector_db.upsert_documents(batch)
                
                # Update progress
                progress = 60.0 + (i / len(documents)) * 30.0
                job.progress = progress
        
        # Store entity embeddings if available
        if all_entities and self.graph_db:
            try:
                # Generate embeddings for entity names
                entity_texts = [f"{e.name}: {e.properties.get('description', '')}" for e in all_entities[:100]]
                entity_embeddings = await self.embedding_generator.generate_embeddings(entity_texts)
                
                # Update entities with embeddings
                for entity, embedding in zip(all_entities[:100], entity_embeddings):
                    entity.embedding = embedding
                    await self.graph_db.update_entity(
                        entity.id,
                        {"embedding": embedding}
                    )
                    
            except Exception as e:
                logger.error(f"Failed to generate entity embeddings: {e}")
        
        job.progress = 100.0
        return len(all_chunks)

    @log_performance
    async def query(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_sources: bool = True,
        use_hyde: Optional[bool] = None,
        doc_type: Optional[str] = None,
    ) -> GeneratedAnswer:
        """
        Query the RAG system for an answer.

        Args:
            query: User query
            top_k: Number of chunks to retrieve
            filters: Metadata filters for retrieval
            include_sources: Whether to include source citations
            use_hyde: Override HyDE setting for this query
            doc_type: Document type for HyDE context (academic, technical, etc.)

        Returns:
            Generated answer with citations

        Raises:
            RAGPipelineError: If query fails
        """
        await self.initialize()
        
        try:
            with LogContext(query_length=len(query)):
                # Retrieve relevant chunks (with optional HyDE enhancement)
                retrieval_kwargs = {
                    "query": query,
                    "top_k": top_k,
                    "filters": filters,
                }
                
                # Add HyDE parameters if supported
                if hasattr(self.retriever, 'retrieve_chunks'):
                    if self.use_hybrid_retrieval:
                        retrieval_kwargs["use_graph"] = self.use_hybrid_retrieval
                    if self.enable_hyde and use_hyde is not None:
                        retrieval_kwargs["use_hyde"] = use_hyde
                    if self.enable_hyde and doc_type:
                        retrieval_kwargs["doc_type"] = doc_type
                    
                    chunks_with_scores = await self.retriever.retrieve_chunks(**retrieval_kwargs)
                else:
                    # Fallback to standard retrieval
                    chunks_with_scores = await self.retriever.retrieve_chunks(**retrieval_kwargs)
                
                if not chunks_with_scores:
                    logger.warning("No relevant chunks found for query")
                    return GeneratedAnswer(
                        query=query,
                        answer="I couldn't find relevant information to answer your question.",
                        citations=[],
                        confidence_score=0.0,
                        processing_time=0.0,
                        model_used=self.answer_generator.model_name,
                    )
                
                # Generate answer
                answer = await self.answer_generator.generate_answer(
                    query=query,
                    chunks=chunks_with_scores,
                )
                
                # Remove sources if not requested
                if not include_sources:
                    answer.citations = []
                
                return answer
                
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise RAGPipelineError(f"Query failed: {str(e)}")

    async def sync_drive_folders(
        self,
        folder_ids: Optional[List[str]] = None,
        process_new: bool = True,
    ) -> Dict[str, Any]:
        """
        Sync PDFs from Google Drive folders.

        Args:
            folder_ids: Specific folders to sync (None for all configured)
            process_new: Whether to process new files immediately

        Returns:
            Sync results
        """
        await self.initialize()
        
        if not self.sync_manager:
            raise RAGPipelineError("Drive sync not configured")
        
        # Sync folders
        sync_results = await self.sync_manager.sync_folders(folder_ids)
        
        if process_new and sync_results["files"]:
            # Process new files
            processing_results = []
            
            for file_info in sync_results["files"]:
                if file_info["status"] in ["new", "updated"]:
                    try:
                        pdf_metadata, chunks = await self.process_pdf_from_drive(
                            file_info["id"]
                        )
                        processing_results.append({
                            "file_id": file_info["id"],
                            "status": "success",
                            "chunks_created": chunks,
                        })
                    except Exception as e:
                        logger.error(f"Failed to process {file_info['name']}: {e}")
                        processing_results.append({
                            "file_id": file_info["id"],
                            "status": "failed",
                            "error": str(e),
                        })
            
            sync_results["processing_results"] = processing_results
        
        return sync_results

    def get_processing_status(self, job_id: str) -> Optional[ProcessingJob]:
        """Get status of a processing job."""
        return self._processing_jobs.get(job_id)

    async def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        await self.initialize()
        
        stats = {
            "total_documents": await self.vector_db.count_documents(),
            "processing_jobs": {
                "total": len(self._processing_jobs),
                "completed": sum(
                    1 for job in self._processing_jobs.values()
                    if job.status == ProcessingStatus.COMPLETED
                ),
                "failed": sum(
                    1 for job in self._processing_jobs.values()
                    if job.status == ProcessingStatus.FAILED
                ),
                "processing": sum(
                    1 for job in self._processing_jobs.values()
                    if job.status == ProcessingStatus.PROCESSING
                ),
            },
        }
        
        if self.sync_manager:
            stats["sync_stats"] = self.sync_manager.get_sync_stats()
        
        # Add graph statistics if available
        if self.graph_db:
            try:
                graph_stats = await self.graph_db.get_statistics()
                stats["graph_stats"] = graph_stats
            except Exception as e:
                logger.error(f"Failed to get graph statistics: {e}")
                stats["graph_stats"] = {"error": str(e)}
        
        return stats

    async def query_with_agent(
        self,
        query: str,
        agent_type: str = "query",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query using an intelligent agent.
        
        Args:
            query: The query to process
            agent_type: Type of agent ("query" or "research")
            context: Additional context for the agent
            
        Returns:
            Agent response with answer and metadata
        """
        await self.initialize()
        
        try:
            from src.agents.query_agent import QueryAgent
            from src.agents.research_agent import ResearchAgent
            from src.agents.tools import AgentTools
            
            # Create agent tools
            tools = AgentTools(
                rag_pipeline=self,
                retriever=self.retriever,
                graph_db=self.graph_db,
            )
            await tools.initialize()
            
            # Create appropriate agent
            if agent_type == "research":
                agent = ResearchAgent(tools=tools)
                result = await agent.conduct_research(query)
            else:
                agent = QueryAgent(tools=tools)
                result = await agent.answer_with_reasoning(query)
            
            # Convert agent result to dictionary
            return {
                "success": True,
                "answer": result.answer if hasattr(result, 'answer') else result.summary,
                "confidence": result.confidence if hasattr(result, 'confidence') else 0.8,
                "sources": result.sources if hasattr(result, 'sources') else result.evidence[:5],
                "entities": result.entities_found if hasattr(result, 'entities_found') else result.entities_analyzed,
                "agent_type": agent_type,
                "metadata": {
                    "key_insights": getattr(result, 'key_insights', []),
                    "recommendations": getattr(result, 'recommendations', []),
                }
            }
            
        except Exception as e:
            logger.error(f"Agent query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_type": agent_type,
            }

    async def consolidate_knowledge_graph(self) -> Dict[str, Any]:
        """
        Consolidate and optimize the knowledge graph.
        
        Returns:
            Consolidation statistics
        """
        if not self.graphiti_builder:
            return {"error": "Graphiti builder not available"}
        
        try:
            stats = await self.graphiti_builder.consolidate_graph()
            logger.info(f"Knowledge graph consolidated: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Graph consolidation failed: {e}")
            return {"error": str(e)}


def create_rag_pipeline(
    use_knowledge_graph: bool = True,
    enable_hyde: bool = False,
) -> RAGPipeline:
    """
    Create a RAG pipeline with default components.
    
    Args:
        use_knowledge_graph: Whether to enable knowledge graph features
        enable_hyde: Whether to enable HyDE query enhancement
        
    Returns:
        Configured RAG pipeline
    """
    settings = get_settings()
    
    return RAGPipeline(
        use_hybrid_retrieval=use_knowledge_graph and settings.enable_knowledge_graph,
        enable_hyde=enable_hyde,
    )