### 🔄 Project Awareness & Context
- **Always read `PLANNING.md`** at the start of a new conversation to understand the project's architecture, goals, style, and constraints.
- **Check `TASK.md`** before starting a new task. If the task isn’t listed, add it with a brief description and today's date.
- **Use consistent naming conventions, file structure, and architecture patterns** as described in `PLANNING.md`.
- **Use venv_ijon** (the virtual environment) whenever executing Python commands, including for unit tests.

### 🧱 Code Structure & Modularity
- **Never create a file longer than 500 lines of code.** If a file approaches this limit, refactor by splitting it into modules or helper files.
- **Organize code into clearly separated modules**, grouped by feature or responsibility.
  For agents this looks like:
    - `agent.py` - Main agent definition and execution logic 
    - `tools.py` - Tool functions used by the agent 
    - `prompts.py` - System prompts
- **Use clear, consistent imports** (prefer relative imports within packages).
- **Use clear, consistent imports** (prefer relative imports within packages).
- **Use python_dotenv and load_env()** for environment variables.

### 🧪 Testing & Reliability
- **Always create Pytest unit tests for new features** (functions, classes, routes, etc).
- **After updating any logic**, check whether existing unit tests need to be updated. If so, do it.
- **Tests should live in a `/tests` folder** mirroring the main app structure.
  - Include at least:
    - 1 test for expected use
    - 1 edge case
    - 1 failure case

### ✅ Task Completion
- **Mark completed tasks in `TASK.md`** immediately after finishing them.
- Add new sub-tasks or TODOs discovered during development to `TASK.md` under a “Discovered During Work” section.

### 📎 Style & Conventions
- **Use Python** as the primary language.
- **Follow PEP8**, use type hints, and format with `black`.
- **Use `pydantic` for data validation**.
- Use `FastAPI` for APIs and `SQLAlchemy` or `SQLModel` for ORM if applicable.
- Write **docstrings for every function** using the Google style:
  ```python
  def example():
      """
      Brief summary.

      Args:
          param1 (type): Description.

      Returns:
          type: Description.
      """
  ```

### 📚 Documentation & Explainability
- **Update `README.md`** when new features are added, dependencies change, or setup steps are modified.
- **Comment non-obvious code** and ensure everything is understandable to a mid-level developer.
- When writing complex logic, **add an inline `# Reason:` comment** explaining the why, not just the what.

### 🧠 AI Behavior Rules
- **Never assume missing context. Ask questions if uncertain.**
- **Never hallucinate libraries or functions** – only use known, verified Python packages.
- **Always confirm file paths and module names** exist before referencing them in code or tests.
- **Never delete or overwrite existing code** unless explicitly instructed to or if part of a task from `TASK.md`.

### 🎯 12-Factor Agent Principles (NEW)
- **Stateless Functions**: Write pure functions - `(input) -> output`, no hidden state
- **Small, Focused Components**: Each class/function does ONE thing well (<100 lines)
- **Explicit Control Flow**: No magic - every step should be visible and debuggable
- **Context Engineering First**: Quality comes from careful prompt/context design, not complex systems
- **Human-in-the-Loop**: Allow humans to validate and improve extractions when quality matters
- **See `/PRPs/quality_extraction_system_v2.md`** for the redesigned architecture
- **See `/docs/12factor_benefits.md`** for detailed benefits and examples
- **Use v2 extractors** in `/extraction/v2/` for new development

### 🚀 Enhanced Prompt System (DEFAULT)
- **USE ENHANCED VERSIONS BY DEFAULT** unless specifically instructed otherwise
- **Enhanced extractors** provide superior quality through:
  - Agent loop architecture with systematic extraction phases
  - Thinking blocks for transparent reasoning processes
  - Academic prose generation with detailed definitions
  - Quality assessment and confidence scoring
  - Structured output with rich metadata
- **Trade-off**: Enhanced prompts take ~2x processing time but provide significantly better extraction quality
- **Simple versions** are available for high-volume, speed-critical tasks when explicitly requested
- **Model preference**: Use **Gemini 2.5 Pro** for enhanced extractions when available
- **Enhanced files locations**:
  - `/extraction/baseline/extractor_enhanced.py` - Enhanced baseline extractor
  - `/extraction/v2/enhancers_enhanced.py` - Enhanced micro-enhancers
  - `/docs/enhanced_prompt_patterns.md` - Documentation of patterns used

### 🗄️ Database Schema Philosophy (CRITICAL)
- **AVOID OVER-ENGINEERING**: Current Neon PostgreSQL schema is well-designed - resist adding complexity
- **User-oriented interfaces**: Added `books` view for human-readable document browsing
- **Enhanced metadata support**: Optional `extraction_metadata` JSONB column for enhanced prompt outputs
- **Schema decisions based on critical analysis**:
  ✅ **Books view** - Genuine UX improvement, low complexity, clear business value
  ✅ **Enhanced metadata** - Captures enhanced prompt value, flexible JSONB storage
  ❌ **Multi-agent schemas** - No evidence needed, current memory tables sufficient
  ❌ **Complex security tables** - No multi-user access patterns identified
  ❌ **Performance optimizations** - No performance issues requiring complex solutions
- **File management**: Use `ksiazki_pdf/` folder for PDF processing with simple monitoring
- **Database utilities**:
  - `add_books_view.py` - Creates user-friendly book browsing interface
  - `add_enhanced_metadata.py` - Adds optional enhanced extraction metadata support
  - `setup_ksiazki_folder.py` - Sets up PDF folder monitoring