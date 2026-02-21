# AGENTS.md - Coding Agent Guidelines

This document provides guidelines for AI coding agents working in the gunnymarc.github.io repository.

## Repository Overview

This is a GitHub Pages blog repository hosting technical articles about AI, machine learning, and data science. The site is available at https://gunnymarc.github.io.

**Repository Type:** Static GitHub Pages site (Markdown-based blog)  
**Primary Content:** Technical blog posts in Markdown format  
**No Build System:** This repository uses GitHub Pages' native Markdown rendering

## Project Structure

```
gunnymarc.github.io/
├── README.md                          # Site description and disclaimer
├── *.md                               # Blog post articles
└── .git/                              # Git repository
```

## Build, Lint, and Test Commands

### Important Note
This repository does NOT have a traditional build system, package.json, or test suite. It relies on GitHub Pages' automatic Jekyll rendering.

### Validation Commands

Since there are no automated tests, validate changes manually:

```bash
# Check git status
git status

# Preview changes locally (if Jekyll is installed)
bundle exec jekyll serve  # Requires local Jekyll setup

# Validate Markdown syntax (using common CLI tools if available)
markdownlint *.md

# Check for broken links (if link checker installed)
markdown-link-check *.md
```

### Git Workflow

```bash
# Check repository status
git status

# Stage changes
git add <filename>

# Commit with descriptive message
git commit -m "Add article about <topic>"

# Push to GitHub (triggers automatic GitHub Pages deployment)
git push origin main
```

## Content Guidelines

### File Naming Conventions

- **Blog Posts:** Use descriptive, hyphen-separated names with spaces
  - Example: `How Do LLMs Work-From Probabilistic Foundations to Intelligent User Segmentation.md`
  - Example: `Understanding How Recommendation Systems Work.md`
  - Example: `What Your RTO Data Isn't Telling You.md`

- **Configuration Files:** Use standard names
  - `README.md` for repository description
  - `AGENTS.md` for agent guidelines (this file)

### Markdown Formatting Standards

Based on existing articles in the repository, follow these conventions:

#### Document Structure

```markdown
# Main Title: Descriptive and SEO-Friendly

*Italicized subtitle or tagline providing context*

---

Introduction paragraph...

## Major Section Heading

Content with proper paragraph breaks...

### Subsection Heading

More detailed content...
```

#### Typography and Formatting

- **Headers:** Use `#` for H1 (title only), `##` for major sections, `###` for subsections
- **Emphasis:** Use `*italic*` for emphasis and `**bold**` for strong emphasis
- **Code Blocks:** Use triple backticks with language identifier
  ```python
  # Example code
  import numpy as np
  ```
- **Inline Code:** Use single backticks for `inline code` references
- **Lists:** Use `-` for unordered lists, numbers for ordered lists
- **Tables:** Use pipe syntax with header row and separator
  ```markdown
  | Column 1 | Column 2 |
  |----------|----------|
  | Value A  | Value B  |
  ```
- **Mathematical Notation:** Use LaTeX syntax with `$$` delimiters
  ```markdown
  $$P(\text{next word} | \text{"The boy went to the"})$$
  ```

#### Code Examples

- Always include language identifier in code blocks (`python`, `bash`, `javascript`, etc.)
- Add comments to explain complex logic
- Use realistic, meaningful variable names
- Include sample output as comments when helpful

Example from existing content:
```python
import numpy as np
from scipy.special import softmax

# Simulating LLM output layer logits
logits = np.array([2.1, 0.3, 3.8, 1.5, 2.9])
tokens = ["Cafe", "Hospital", "Playground", "Park", "School"]

# Convert to probabilities via softmax
probabilities = softmax(logits)

for token, prob in zip(tokens, probabilities):
    print(f"{token}: {prob:.3f}")
# Output: Playground has highest probability at 0.467
```

#### Content Style

- **Tone:** Technical but accessible; explain complex concepts clearly
- **Audience:** Data scientists, ML engineers, and technical professionals
- **Length:** Comprehensive articles (250-350+ lines)
- **Structure:** Use narrative flow with clear transitions between sections
- **References:** Include footnotes using `[^1]` notation when citing sources

### Writing Best Practices

1. **Start with a hook:** Begin articles with engaging questions or insights
2. **Use concrete examples:** Include real code, data, and scenarios
3. **Visual aids:** Use tables, code blocks, and mathematical notation
4. **Clear explanations:** Break down complex topics into digestible sections
5. **Practical applications:** Show real-world use cases and implementations

## Git Commit Message Conventions

Based on repository history, commit messages are clear and descriptive:

```bash
# Good examples
git commit -m "Add article about LLM training pipeline"
git commit -m "Update README with site description"
git commit -m "Rename article for clarity and SEO"

# Avoid vague messages
git commit -m "Update"
git commit -m "Fix"
```

## Working with This Repository

### Adding a New Blog Post

1. Create a new `.md` file with descriptive name
2. Follow the markdown structure template above
3. Include proper frontmatter (title, subtitle, sections)
4. Add code examples with proper syntax highlighting
5. Validate markdown syntax if possible
6. Commit with descriptive message
7. Push to main branch (triggers GitHub Pages build)

### Editing Existing Content

1. Read the entire article first to understand context
2. Maintain consistent voice and style with existing content
3. Preserve existing formatting conventions
4. Test code examples if making technical changes
5. Commit changes with clear description of what was modified

### Important Constraints

- **No emojis:** Keep content professional and text-based
- **No placeholder content:** All code examples should be functional
- **Preserve structure:** Maintain existing file organization
- **GitHub Pages limitations:** Remember that GitHub Pages uses Jekyll by default
  - Markdown files are automatically rendered
  - No custom build process
  - Changes are live shortly after push to main

## Disclaimer

All content should align with the site's disclaimer in README.md:
- Published in good faith
- For general information purposes
- No warranties about completeness or accuracy
- Users act at their own risk

## About the Author

The site is maintained by a U.S. Marine and Data Scientist. Content reflects expertise in:
- Machine Learning and AI
- Large Language Models
- Recommendation Systems
- Data Science and Analytics

When contributing, maintain the technical depth and practical focus evident in existing articles.
