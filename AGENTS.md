# Agent Instructions for gunnymarc.github.io

Personal blog/portfolio built with Jekyll (fork of [academicpages](https://github.com/academicpages/academicpages.github.io) / Minimal Mistakes) and served via GitHub Pages.
Content is long-form technical articles (LLMs, Data Science) in Markdown.

## 1. Build, Lint & Test Commands

**Stack:** Ruby 3.1+, Jekyll 3.9 (via `github-pages` gem), academicpages theme (fork-based, not a gem).
**Deployment:** Push to `main` — GitHub Pages builds and deploys automatically.

### Global Build & Serve

| Action | Command |
| :--- | :--- |
| **Install deps** | `bundle install` |
| **Serve locally** | `bundle exec jekyll serve --livereload` (http://localhost:4000) |
| **Build only** | `bundle exec jekyll build` (output in `_site/`) |
| **Clean** | `bundle exec jekyll clean` |

*Note: Use Ruby 3.1 for local builds (brew install ruby@3.1). The system Ruby (2.6.x) is too old.*

### Testing an Article

1. Build: `bundle exec jekyll build`
2. Check the output in `_site/posts/YYYY/MM/slug/`
3. Verify image paths: `grep -r "assets/images" _site/posts/`

## 2. Code Style Guidelines

### File Naming & Location Conventions

- **Articles (Blog Posts):** Go in `_posts/` with date prefix — `YYYY-MM-DD-slug.md`
- **URLs:** Auto-generated from slug: `/posts/YYYY/MM/slug/`
- **Assets:** Images in `assets/images/`, CSS in `assets/css/`
- **Static Pages:** Go in `_pages/` with `permalink:` in front matter
- **Navigation:** Edit `_data/navigation.yml`
- **Old URLs:** Use `redirect_from` in post front matter to redirect from previous URLs

### Formatting & Structure (Markdown)

- **Front Matter:** Every article requires YAML front matter with at minimum:
  ```yaml
  ---
  title: "Article Title"
  date: YYYY-MM-DD
  permalink: /posts/YYYY/MM/slug/
  tags:
    - tag1
    - tag2
  ---
  ```
- **Title:** Do NOT include `# Title` (H1) in content — the theme renders it from front matter.
- **Subtitle:** Optional `*Italic subtitle*` or `> Blockquote subtitle`.
- **Table of Contents:** Optional but preferred. Links format: `[Section Name](#lowercase-hyphenated-name)`.
- **Headings:** Use `##` (H2) for major sections, `###` (H3) for subsections. Never skip header levels.
- **Separators:** Use `---` between major sections.

### Styling Elements

- **Code Blocks:** Triple backticks with language tags (` ```python `, ` ```bash `).
- **Math:** `$$` for block math, `$` for inline math.
- **Tables:** GitHub Flavored Markdown pipe tables with left-aligned headers (`:---`).
- **Lists:** `-` for unordered lists, `1.` for ordered lists.
- **Emphasis:** `**bold**` for key terms/definitions, `*italic*` for nuances or book titles.
- **Footnotes:** `[^1]` inline, `[^1]: text` at the point of use.

### Theme Customization

- **CSS Overrides:** Edit `assets/css/main.scss` (add overrides AFTER the `@import` block).
- **Theme variants:** Set `site_theme` in `_config.yml` to `default`, `air`, `sunrise`, `mint`, `dirt`, or `contrast`.
- **Layouts:** Located in `_layouts/` — `single.html` is the main layout for posts and pages.

## 3. Repository Structure

```text
/
├── _config.yml              # Jekyll configuration
├── _data/                   # Navigation, UI text, etc.
├── _includes/               # Reusable HTML components (theme)
├── _layouts/                # HTML layout templates (theme)
├── _pages/                  # Static pages (about, archive, 404)
├── _posts/                  # Blog posts (YYYY-MM-DD-slug.md)
├── _sass/                   # SCSS stylesheets (theme)
├── assets/                  # CSS, JS, images, fonts
├── files/                   # Downloads (PDFs, etc.)
├── Gemfile                  # Ruby dependencies
├── README.md                # Article index (update when adding articles!)
├── AGENTS.md                # This agent instructions file
└── .gitignore
```

## 4. Git & Workflow Conventions

- **Branching:** `main` is production. Use `feature/name`, `fix/name`, or `docs/name`.
- **Commits:** Follow conventional commits (e.g., `feat: add LLM temperature article`, `fix: correct typo`).
- **Homepage:** When a new article is added, **you must** add a bullet to the Articles section in `README.md` in reverse chronological order:
  `- *Month Year* — [Article Title](https://gunnymarc.github.io/posts/YYYY/MM/slug/)`
- **Redirects:** If changing a post's URL, add `redirect_from:` to the front matter pointing to the old URL.

## 5. Constraints

- Do NOT add npm, Webpack, or complex build tooling.
- Do NOT commit secrets, API keys, or credentials.
- All files must be UTF-8 encoded.
- Use `assets/images/` for images, not `images/`.

## 6. AI Agent Rules

There are no global `.cursorrules` or `.github/copilot-instructions.md` files in this repository. This `AGENTS.md` is the single source of truth for all LLMs.

- Always treat `AGENTS.md` as authoritative.
- Read existing articles to mimic the author's precise tone, mathematical rigor, and formatting style before writing or editing.
- Never summarize your actions after writing to a file unless asked.

## 7. Troubleshooting

| Error | Fix |
| :--- | :--- |
| **Liquid Exception** | Check `{{ }}` and `{% %}` syntax in Markdown/SCSS files. |
| **YAML Parse Error** | Validate front matter delimiters (`---`). Missing title/date/permalinks. |
| **Images Not Loading** | Verify path is relative to site root. Use `assets/images/` prefix. |
| **Encoding Error** | Ensure UTF-8. |
| **Stale Content** | Run `bundle exec jekyll clean` then rebuild. |
| **Old URL 404s** | Add `redirect_from: /old-url` to post front matter. |

## 8. Local Ruby Setup

Use Ruby 3.1 (not system Ruby 2.6):

```bash
brew install ruby@3.1
export PATH="/opt/homebrew/opt/ruby@3.1/bin:$PATH"
gem install bundler
bundle install
```
