"""
Create sample PDFs for testing the extraction system.

Since we can't easily access the existing PDFs, let's create simple test PDFs
that represent each category.
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
import json


class SamplePDFGenerator:
    """Generate sample PDFs for each test category."""
    
    def __init__(self, output_dir="test_documents"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def create_technical_manual(self):
        """Create a technical manual PDF."""
        filename = self.output_dir / "technical_manual.pdf"
        doc = SimpleDocTemplate(str(filename), pagesize=letter)
        
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph("Python API Reference Manual", styles['Title']))
        story.append(Spacer(1, 0.5*inch))
        
        # Content
        content = [
            ("1. Introduction", """
            This manual provides a comprehensive reference for the Python REST API framework.
            The framework enables developers to quickly build scalable web services with 
            minimal boilerplate code. It includes features for authentication, rate limiting,
            and automatic documentation generation.
            """),
            
            ("2. Installation", """
            To install the framework, use pip:
            
            pip install python-api-framework
            
            Requirements:
            - Python 3.8 or higher
            - asyncio support
            - PostgreSQL 12+ for database operations
            """),
            
            ("3. Basic Usage", """
            Here's a simple example of creating an API endpoint:
            
            from api_framework import APIRouter, Response
            
            router = APIRouter()
            
            @router.get("/users/{user_id}")
            async def get_user(user_id: int) -> Response:
                user = await fetch_user(user_id)
                return Response(data=user, status=200)
            
            The framework automatically handles JSON serialization, error responses,
            and request validation based on type hints.
            """),
            
            ("4. Configuration", """
            Configuration is managed through environment variables or a config file:
            
            API_HOST: Server host (default: localhost)
            API_PORT: Server port (default: 8000)
            DATABASE_URL: PostgreSQL connection string
            JWT_SECRET: Secret key for JWT token generation
            RATE_LIMIT: Requests per minute (default: 100)
            """),
            
            ("5. Error Handling", """
            The framework provides built-in error handling with proper HTTP status codes:
            
            - 400: Bad Request - Invalid input parameters
            - 401: Unauthorized - Missing or invalid authentication
            - 403: Forbidden - Insufficient permissions
            - 404: Not Found - Resource does not exist
            - 500: Internal Server Error - Unexpected server error
            
            Custom error handlers can be registered for specific exceptions.
            """)
        ]
        
        for title, text in content:
            story.append(Paragraph(title, styles['Heading2']))
            story.append(Paragraph(text, styles['BodyText']))
            story.append(Spacer(1, 0.2*inch))
        
        doc.build(story)
        
        # Create metadata
        self._create_metadata("technical_manual", "technical", 
                            "Python API Reference Manual", 
                            ["structured", "code_examples", "specifications", "configuration"])
        
        return filename
    
    def create_academic_paper(self):
        """Create an academic paper PDF."""
        filename = self.output_dir / "academic_paper.pdf"
        doc = SimpleDocTemplate(str(filename), pagesize=letter)
        
        styles = getSampleStyleSheet()
        story = []
        
        # Title and Abstract
        story.append(Paragraph("Machine Learning Applications in Natural Language Processing: A Systematic Review", styles['Title']))
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("Abstract", styles['Heading2']))
        story.append(Paragraph("""
        This paper presents a comprehensive systematic review of machine learning applications
        in natural language processing (NLP) from 2019 to 2024. We analyzed 156 peer-reviewed
        articles to identify trends, methodologies, and performance metrics. Our findings
        indicate that transformer-based models have dominated the field, achieving state-of-the-art
        results in 78% of surveyed tasks. We also identify key challenges and future research
        directions in multilingual NLP and low-resource languages.
        """, styles['BodyText']))
        story.append(Spacer(1, 0.3*inch))
        
        # Main content
        content = [
            ("1. Introduction", """
            Natural Language Processing (NLP) has experienced rapid advancement with the
            integration of machine learning techniques. The ability to automatically process
            and understand human language has applications ranging from translation to
            sentiment analysis. This systematic review examines the current state of ML
            applications in NLP, focusing on methodological trends and performance benchmarks.
            
            Previous surveys (Chen et al., 2020; Kumar & Singh, 2021) have examined specific
            aspects of NLP, but a comprehensive analysis of recent developments is lacking.
            Our contribution is threefold: (1) we provide a taxonomy of ML approaches in NLP,
            (2) we analyze performance metrics across different tasks, and (3) we identify
            emerging trends and research gaps.
            """),
            
            ("2. Methodology", """
            We followed the PRISMA guidelines for systematic reviews. Our search strategy
            included five major databases: IEEE Xplore, ACM Digital Library, SpringerLink,
            ScienceDirect, and arXiv. Search terms included combinations of "machine learning",
            "deep learning", "natural language processing", and specific task names.
            
            Inclusion criteria:
            - Published between January 2019 and December 2024
            - Peer-reviewed conference or journal articles
            - Empirical evaluation on standard benchmarks
            - English language publications
            
            From an initial pool of 1,247 papers, we selected 156 after applying our criteria
            and removing duplicates. Each paper was coded for ML methodology, NLP task,
            dataset used, and performance metrics.
            """),
            
            ("3. Results", """
            Our analysis reveals several key findings:
            
            3.1 Dominant Architectures
            Transformer-based models appeared in 122 papers (78.2%), with BERT variants
            being the most common (45 papers). GPT-style models were used in 38 papers,
            primarily for generation tasks. Traditional RNNs and CNNs appeared in only
            12 papers (7.7%), mostly as baselines.
            
            3.2 Task Distribution
            The most studied tasks were:
            - Text Classification: 42 papers (26.9%)
            - Machine Translation: 31 papers (19.9%)
            - Named Entity Recognition: 28 papers (17.9%)
            - Question Answering: 23 papers (14.7%)
            - Text Generation: 20 papers (12.8%)
            
            3.3 Performance Trends
            Average performance improvements over baseline methods:
            - BERT-based models: +12.3% F1 score
            - GPT variants: +8.7% BLEU score
            - Multi-task learning: +6.2% across tasks
            """),
            
            ("4. Discussion", """
            The dominance of transformer architectures reflects their superior performance
            across diverse NLP tasks. However, this trend raises concerns about computational
            requirements and accessibility for researchers with limited resources.
            
            We identified several research gaps:
            1. Limited work on low-resource languages (only 11 papers)
            2. Insufficient attention to model interpretability
            3. Few studies on energy efficiency of large models
            
            These gaps suggest important directions for future research, particularly
            as NLP systems are deployed in resource-constrained environments.
            """),
            
            ("References", """
            Chen, L., Zhang, W., & Liu, Y. (2020). Deep learning for NLP: A survey of
            recent advances. ACM Computing Surveys, 53(5), 1-40.
            
            Kumar, S., & Singh, P. (2021). Transformer models in natural language
            understanding: A comprehensive review. IEEE Transactions on Neural Networks
            and Learning Systems, 32(8), 3421-3440.
            
            Williams, R., Johnson, K., & Brown, M. (2022). Benchmarking neural
            architectures for multilingual NLP. Proceedings of ACL 2022, 234-248.
            """)
        ]
        
        for title, text in content:
            story.append(Paragraph(title, styles['Heading2']))
            story.append(Paragraph(text, styles['BodyText']))
            story.append(Spacer(1, 0.2*inch))
        
        doc.build(story)
        
        self._create_metadata("academic_paper", "academic",
                            "Machine Learning Applications in NLP: A Systematic Review",
                            ["research", "citations", "methodology", "systematic_review"])
        
        return filename
    
    def create_business_book(self):
        """Create a business book chapter PDF."""
        filename = self.output_dir / "business_book.pdf"
        doc = SimpleDocTemplate(str(filename), pagesize=letter)
        
        styles = getSampleStyleSheet()
        story = []
        
        # Chapter title
        story.append(Paragraph("Chapter 7: The Innovation Paradox", styles['Title']))
        story.append(Spacer(1, 0.5*inch))
        
        content = [
            ("", """
            In the summer of 2007, Nokia commanded nearly 50% of the global mobile phone
            market. Their executives dismissed the iPhone as a niche product for tech
            enthusiasts. "It doesn't even have a keyboard," one senior manager reportedly
            scoffed. Within five years, Nokia's mobile division was sold to Microsoft
            for a fraction of its former value. This story illustrates what I call the
            Innovation Paradox: the very factors that make a company successful can
            become the barriers to its future innovation.
            """),
            
            ("The Comfort of Success", """
            Success breeds complacency. When quarterly earnings exceed expectations and
            market share grows steadily, questioning the fundamental business model seems
            not just unnecessary but foolish. "If it ain't broke, don't fix it" becomes
            the unofficial company motto. 
            
            I witnessed this firsthand at Kodak in the 1990s. Despite inventing the
            digital camera in 1975, Kodak's leadership couldn't imagine a world without
            film. The profit margins on film were extraordinary – why cannibalize that
            business? This thinking, logical in the short term, proved catastrophic in
            the long term.
            
            The paradox deepens when we consider that innovative companies often have
            the resources, talent, and market position to adapt. What they lack is the
            organizational will to disrupt themselves.
            """),
            
            ("Breaking Free from the Paradox", """
            How can established companies escape the Innovation Paradox? Through my work
            with Fortune 500 companies, I've identified three critical strategies:
            
            1. Create a Culture of Constructive Paranoia
            Andy Grove famously said, "Only the paranoid survive." Leaders must cultivate
            a healthy skepticism about current success. At Amazon, Jeff Bezos insists it's
            always "Day 1" – the moment you think you've figured it out, decline begins.
            
            2. Establish Innovation Safe Zones
            3M's "15% time" policy, which led to the creation of Post-it Notes, demonstrates
            the power of protected innovation space. These zones must be truly autonomous,
            with different metrics, timelines, and even physical separation from the core
            business.
            
            3. Embrace Controlled Cannibalization
            Netflix's transition from DVD-by-mail to streaming exemplifies controlled
            self-disruption. Despite a profitable DVD business, Reed Hastings saw the
            digital future and pivoted before competitors forced the change.
            """),
            
            ("Case Study: Microsoft's Reinvention", """
            Microsoft's transformation under Satya Nadella provides a masterclass in
            escaping the Innovation Paradox. When Nadella became CEO in 2014, Microsoft
            was seen as a declining giant, clinging to Windows and Office while missing
            mobile and cloud computing.
            
            Nadella's first major decision was symbolic but powerful: he held up an iPhone
            at a company meeting and demonstrated Microsoft Office running on it. This
            shocked employees accustomed to viewing Apple as the enemy. The message was
            clear: Microsoft would meet customers where they were, not where Microsoft
            wanted them to be.
            
            The shift to cloud-first strategy required cannibalizing profitable on-premise
            software sales. Wall Street was skeptical. But by 2020, Azure had become a
            $50 billion business, and Microsoft's market cap exceeded $1 trillion.
            
            Key lessons from Microsoft's transformation:
            - Leadership must model the change they seek
            - Sacred cows must be questioned (even Windows)
            - Customer needs trump internal politics
            - Long-term vision requires short-term sacrifice
            """),
            
            ("Action Steps for Leaders", """
            To apply these insights in your organization:
            
            1. Conduct a "Disruption Audit"
            List your most profitable products/services. For each, identify three ways
            a startup might attack them. If you can't find three, you're not thinking
            hard enough.
            
            2. Create a Shadow Board
            Assemble a group of high-potential employees under 35. Task them with
            presenting to the board quarterly on threats and opportunities the senior
            leadership might miss.
            
            3. Set Innovation Metrics
            Traditional metrics reward optimization of existing business. Create new KPIs
            that reward learning, experimentation, and intelligent failure.
            
            4. Partner with Potential Disruptors
            Instead of dismissing startups as irrelevant, engage with them. Acquisition,
            investment, or partnership can inject innovative thinking into your organization.
            
            Remember: The Innovation Paradox is not a death sentence but a challenge.
            Companies that recognize and actively counter it can reinvent themselves
            repeatedly. Those that don't become cautionary tales in business school
            case studies. Which will your company be?
            """)
        ]
        
        for title, text in content:
            if title:
                story.append(Paragraph(title, styles['Heading2']))
            story.append(Paragraph(text, styles['BodyText']))
            story.append(Spacer(1, 0.2*inch))
        
        doc.build(story)
        
        self._create_metadata("business_book", "business",
                            "The Innovation Paradox - Chapter 7",
                            ["business_strategy", "case_studies", "innovation", "leadership"])
        
        return filename
    
    def create_tutorial_guide(self):
        """Create a tutorial/how-to guide PDF."""
        filename = self.output_dir / "tutorial_guide.pdf"
        doc = SimpleDocTemplate(str(filename), pagesize=letter)
        
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph("Complete Guide to Building Your First Web Application", styles['Title']))
        story.append(Spacer(1, 0.5*inch))
        
        content = [
            ("Introduction", """
            Building your first web application can seem daunting, but with the right
            approach, you'll have a working app in just a few hours. This guide will
            walk you through creating a simple task management application using modern
            web technologies. No prior experience required – just enthusiasm to learn!
            """),
            
            ("What You'll Need", """
            Before we begin, make sure you have the following installed:
            
            • A text editor (VS Code recommended - free from code.visualstudio.com)
            • Node.js (download from nodejs.org - choose the LTS version)
            • A web browser (Chrome or Firefox work best for development)
            • Basic familiarity with using a computer terminal/command prompt
            
            Don't worry if you've never used these tools before. We'll guide you through
            each step.
            """),
            
            ("Step 1: Setting Up Your Project", """
            First, let's create a new directory for your project:
            
            1. Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux)
            2. Navigate to where you want to create your project:
               cd ~/Desktop
            3. Create a new directory:
               mkdir my-first-web-app
            4. Enter the directory:
               cd my-first-web-app
            5. Initialize a new Node.js project:
               npm init -y
            
            Great! You've just created the foundation for your web application. The npm init
            command created a package.json file, which tracks your project's dependencies.
            """),
            
            ("Step 2: Creating Your First HTML Page", """
            Now let's create the structure of your web application:
            
            1. In your text editor, create a new file called index.html
            2. Add the following HTML structure:
            
            The HTML should include:
            - A DOCTYPE declaration for HTML5
            - A head section with meta tags for charset and viewport
            - A title element saying "My Task Manager"
            - A link to a CSS file called styles.css
            - A body with a container div containing:
              - An h1 heading with "My Task Manager"
              - An input field for entering new tasks
              - A button to add tasks
              - An unordered list to display tasks
            - A script tag linking to app.js
            
            This HTML creates the basic structure: a title, an input field for new tasks,
            a button to add them, and a list to display them.
            """),
            
            ("Step 3: Styling Your Application", """
            Let's make your app look professional with some CSS:
            
            1. Create a new file called styles.css
            2. Add styles for the following elements:
            
            For the body:
            - Use Arial or sans-serif font
            - Light gray background color
            - Remove default margins and add padding
            
            For the container:
            - Maximum width of 600px
            - Center it with auto margins
            - White background
            - Add padding and rounded corners
            - Include a subtle shadow
            
            For the heading:
            - Dark gray color
            - Center alignment
            
            For the task input area:
            - Use flexbox layout
            - Add bottom margin
            
            For the input field:
            - Make it flexible to fill available space
            - Add padding and borders
            - Round the left corners
            
            For buttons:
            - Green background with white text
            - Remove default borders
            - Round the right corners
            - Change cursor to pointer on hover
            - Darken on hover for feedback
            """),
            
            ("Step 4: Adding Functionality with JavaScript", """
            Now for the exciting part - making your app interactive!
            
            1. Create a file called app.js
            2. Add JavaScript code that includes:
            
            Core components:
            - An array to store tasks
            - A function to add new tasks
            - A function to display all tasks
            - Functions to toggle and delete tasks
            
            The addTask function should:
            - Get the input element by its ID
            - Extract and trim the text value
            - Validate that text was entered
            - Create a task object with id, text, and completed status
            - Add the task to the array
            - Clear the input field
            - Update the display
            
            The displayTasks function should:
            - Get the task list element
            - Clear existing content
            - Loop through all tasks
            - Create list items with the task text
            - Add buttons for completing and deleting tasks
            - Apply styling for completed tasks
            
            Task objects should have:
            - A unique ID (using timestamp)
            - The task text
            - A completed boolean flag
            """),
            
            ("Step 5: Testing Your Application", """
            Time to see your creation in action!
            
            1. Save all three files (index.html, styles.css, app.js)
            2. Double-click on index.html to open it in your browser
            3. Try adding some tasks
            4. Click the check button to mark tasks as complete
            5. Click the X button to delete tasks
            
            Troubleshooting Tips:
            • If nothing happens when you click buttons, check the browser console
              (Press F12 and click Console tab) for error messages
            • Make sure all file names match exactly (case-sensitive)
            • Verify that all three files are in the same directory
            """),
            
            ("Next Steps", """
            Congratulations! You've built your first web application. Here are some ideas
            to enhance it further:
            
            1. Add local storage to save tasks between sessions
            2. Include due dates for tasks
            3. Create categories or tags
            4. Add a search/filter function
            5. Make it responsive for mobile devices
            
            Resources for continued learning:
            • MDN Web Docs (developer.mozilla.org) - Comprehensive web development reference
            • freeCodeCamp (freecodecamp.org) - Free coding courses
            • JavaScript30 (javascript30.com) - 30 day vanilla JS challenge
            
            Remember: Every expert was once a beginner. Keep practicing, stay curious,
            and don't be afraid to experiment. Happy coding!
            """)
        ]
        
        for title, text in content:
            story.append(Paragraph(title, styles['Heading2']))
            story.append(Paragraph(text, styles['BodyText']))
            story.append(Spacer(1, 0.2*inch))
        
        doc.build(story)
        
        self._create_metadata("tutorial_guide", "tutorial",
                            "Complete Guide to Building Your First Web Application",
                            ["step_by_step", "examples", "beginner_friendly", "web_development"])
        
        return filename
    
    def create_historical_text(self):
        """Create a historical text PDF."""
        filename = self.output_dir / "historical_text.pdf"
        doc = SimpleDocTemplate(str(filename), pagesize=letter)
        
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph("The Dawn of the Information Age: Computing in the 1940s", styles['Title']))
        story.append(Spacer(1, 0.5*inch))
        
        content = [
            ("Preface", """
            The decade of the 1940s witnessed a transformation that would reshape human
            civilization as profoundly as the Industrial Revolution. In laboratories and
            research facilities across the Allied nations, scientists and engineers were
            creating the first electronic digital computers. This account, drawn from
            interviews with surviving pioneers and declassified wartime documents, reveals
            how the urgency of global conflict accelerated technological progress that
            might otherwise have taken decades to achieve.
            """),
            
            ("Chapter 1: The Catalyst of War", """
            In the spring of 1943, as Allied forces prepared for the invasion of Italy,
            a different kind of battle was being waged in the basement of the Moore School
            of Electrical Engineering at the University of Pennsylvania. John Mauchly and
            J. Presper Eckert, along with a team of engineers, were racing against time
            to complete a machine that would revolutionize warfare – and ultimately, the
            world. The Electronic Numerical Integrator and Computer, known as ENIAC, was
            born from desperate necessity.
            
            The United States Army's Ballistic Research Laboratory faced an overwhelming
            challenge. Each new artillery piece required firing tables containing thousands
            of trajectory calculations. A skilled mathematician with a mechanical calculator
            could compute a single trajectory in twenty hours. The Army needed thousands
            of trajectories for each weapon. The mathematics of death had become a
            bottleneck in the machinery of war.
            
            "We knew that every day of delay meant American soldiers were firing artillery
            with imprecise calculations," recalled Herman Goldstine, the Army liaison to
            the ENIAC project, in a 1972 interview. "The pressure was immense. We worked
            eighteen-hour days, seven days a week. There was no concept of weekends or
            holidays. The war didn't take breaks."
            """),
            
            ("Chapter 2: The Architecture of Innovation", """
            ENIAC's design represented a radical departure from all previous calculating
            machines. Where mechanical calculators processed one operation at a time,
            ENIAC could perform 5,000 additions per second. Its 17,468 vacuum tubes,
            70,000 resistors, and 10,000 capacitors filled a room 30 by 50 feet. The
            machine consumed 150 kilowatts of power – enough to dim the lights in the
            surrounding neighborhood when it was switched on.
            
            Yet ENIAC's true innovation lay not in its size or speed, but in its
            programmability. Unlike earlier fixed-purpose calculators, ENIAC could be
            reconfigured to solve different problems. This flexibility came at a cost:
            programming ENIAC meant physically rewiring the machine, a process that could
            take days.
            
            The women who programmed ENIAC – Kay McNulty, Betty Jennings, Betty Snyder,
            Marlyn Meltzer, Fran Bilas, and Ruth Lichterman – were mathematicians recruited
            from the Army's corps of human "computers." Their work, largely unrecognized
            for decades, established many fundamental principles of programming.
            
            "We had no programming languages, no operating systems, nothing," Kay McNulty
            Mauchly Antonelli later recalled. "We had to think like the machine. Every
            program was an act of invention."
            """),
            
            ("Chapter 3: Parallel Developments", """
            While ENIAC captured public imagination when its existence was revealed in 1946,
            it was not the only electronic computer under development during the war years.
            In Britain, the Government Code and Cypher School at Bletchley Park had
            constructed Colossus, a special-purpose electronic computer designed to break
            German encryption.
            
            The secrecy surrounding Colossus was absolute. While ENIAC's creators would
            receive recognition and acclaim, Colossus's designers – notably Tommy Flowers
            and Max Newman – would remain anonymous for nearly three decades. The British
            government ordered the destruction of most Colossus machines after the war,
            and participants were bound by the Official Secrets Act.
            
            Meanwhile, in neutral Switzerland, Konrad Zuse continued developing his Z-series
            computers in relative isolation. His Z3, completed in 1941, was arguably the
            world's first programmable, fully automatic digital computer. Built with
            telephone relays instead of vacuum tubes, it was slower than its electronic
            contemporaries but demonstrated that the fundamental principles of digital
            computation transcended specific technologies.
            """),
            
            ("Chapter 4: The Stored Program Revolution", """
            Even as ENIAC performed its first calculations, its creators recognized its
            limitations. The lengthy rewiring process required for each new program was
            impractical for a general-purpose computer. The solution came from an unlikely
            collaboration between mathematician John von Neumann and the ENIAC team.
            
            Von Neumann's "First Draft of a Report on the EDVAC" (Electronic Discrete
            Variable Automatic Computer), circulated in June 1945, proposed storing
            programs in the computer's memory alongside data. This concept – the stored
            program architecture – would become the foundation of all modern computers.
            
            The report sparked controversy over attribution that persists to this day.
            Eckert and Mauchly claimed they had independently developed the concept.
            Von Neumann's name on the widely-circulated report, however, led to the
            architecture being dubbed the "von Neumann architecture."
            
            "The tragedy was that what should have been a collaborative triumph became
            a bitter dispute over credit," observed historian Nancy Stern. "The stored
            program concept was probably an inevitable convergence of ideas, but the
            conflict fractured the original ENIAC team and scattered their talents to
            different institutions."
            """),
            
            ("Chapter 5: From Laboratory to Industry", """
            The transition from wartime research to commercial computing began even before
            the war's end. Eckert and Mauchly, frustrated by disputes over patent rights
            at the University of Pennsylvania, formed the Electronic Controls Company in
            1946, later renamed the Eckert-Mauchly Computer Corporation. Their goal was
            ambitious: to build computers for business and scientific applications.
            
            The challenges were formidable. Vacuum tubes, the computer's essential
            components, were unreliable. ENIAC's tubes failed at an average rate of one
            every two days. For commercial viability, computers needed to run for weeks
            or months without failure. The solution required not just better tubes, but
            fundamental advances in error detection and correction.
            
            Financial challenges proved equally daunting. Building computers required
            enormous capital investment with uncertain returns. The Eckert-Mauchly
            Computer Corporation struggled until Remington Rand acquired it in 1950.
            This pattern – innovative startups absorbed by established corporations –
            would become a recurring theme in the computer industry.
            """),
            
            ("Epilogue: The Future They Imagined", """
            In 1949, Popular Mechanics made a bold prediction: "Computers in the future
            may weigh no more than 1.5 tons." This forecast, ridiculed today for its
            lack of imagination, actually represented remarkable optimism. ENIAC weighed
            30 tons. To imagine a computer weighing merely 1.5 tons was to envision a
            twentyfold improvement in just a few years.
            
            The pioneers of the 1940s could hardly imagine computers that would fit in
            a pocket, yet they glimpsed the transformation their work would bring. In a
            1948 letter, Alan Turing wrote: "The machine is not a mere tool like a
            hammer or a printing press. It is an amplifier of intelligence itself. Its
            impact will be felt in every sphere of human endeavor."
            
            As we stand in the digital age these visionaries created, their legacy
            reminds us that revolutionary change often begins in modest circumstances:
            a basement laboratory, a wartime project, a small group of dedicated
            individuals pursuing an idea whose time has come. The information age was
            born not in a single moment of inspiration, but through years of persistent
            effort, collaboration, and the courage to imagine a different future.
            """)
        ]
        
        for title, text in content:
            if title:
                story.append(Paragraph(title, styles['Heading2']))
            story.append(Paragraph(text, styles['BodyText']))
            story.append(Spacer(1, 0.2*inch))
        
        doc.build(story)
        
        self._create_metadata("historical_text", "historical",
                            "The Dawn of the Information Age: Computing in the 1940s",
                            ["historical_narrative", "technology_history", "1940s", "computing_pioneers"])
        
        return filename
    
    def _create_metadata(self, category, doc_type, title, characteristics):
        """Create metadata JSON file for a PDF."""
        metadata = {
            "filename": f"{category}.pdf",
            "document_type": doc_type,
            "title": title,
            "source": "Generated for testing",
            "page_count": 5,  # Approximate
            "language": "en",
            "characteristics": characteristics,
            "expected_extraction_difficulty": "medium",
            "notes": f"Synthetic test document for {category} category"
        }
        
        metadata_path = self.output_dir / f"{category}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def generate_all(self):
        """Generate all test PDFs."""
        print("🔧 Generating sample PDFs for testing...")
        
        try:
            # Install reportlab if needed
            import reportlab
        except ImportError:
            print("📦 Installing reportlab for PDF generation...")
            import subprocess
            subprocess.check_call(["pip", "install", "reportlab"])
            import reportlab
        
        pdfs = [
            ("Technical Manual", self.create_technical_manual),
            ("Academic Paper", self.create_academic_paper),
            ("Business Book", self.create_business_book),
            ("Tutorial Guide", self.create_tutorial_guide),
            ("Historical Text", self.create_historical_text)
        ]
        
        for name, func in pdfs:
            try:
                filename = func()
                print(f"✅ Created {name}: {filename}")
            except Exception as e:
                print(f"❌ Failed to create {name}: {str(e)}")
        
        print("\n✅ Sample PDFs created successfully!")
        print("Next steps:")
        print("1. Run the baseline evaluation script")
        print("2. Complete manual validation when prompted")
        print("3. Review the generated reports")


if __name__ == "__main__":
    generator = SamplePDFGenerator()
    generator.generate_all()