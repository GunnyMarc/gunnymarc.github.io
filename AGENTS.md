# AGENTS.md — gunnymarc.github.io

Personal blog/portfolio (Marc Buraczynski) — Jekyll static site using the Chirpy theme, deployed via GitHub Pages on push to `main`.

## Commands

| Action | Command |
| :--- | :--- |
| Install deps | `bundle install` (use Ruby 3.2: `export PATH="/opt/homebrew/opt/ruby@3.2/bin:$PATH"`) |
| Serve locally | `bundle exec jekyll serve --livereload` (http://localhost:4000) |
| Build | `bundle exec jekyll build` (output in `_site/`) |
| Clean cache | `bundle exec jekyll clean` |

**Ruby 3.2+ required** (system Ruby 2.6 is too old, use Homebrew `ruby@3.2`). No test suite — "testing" means `jekyll build` then inspect `_site/posts/YYYY/MM/slug/`.

## Architecture

- **Theme is gemmed (Chirpy 7.x).** `_layouts/`, `_includes/`, `_sass/` are provided by the `jekyll-theme-chirpy` gem. Do not edit these directly; override in-repo files take precedence.
- **CSS overrides** go in `_sass/override/` or `_sass/addon/` (Chirpy conventions). Custom SCSS files are imported automatically by the theme.
- **No collections.** The Minimal Mistakes collections (portfolio, publications, talks, teaching) have been removed.
- **Per-post `permalink:` is required in front matter.** Always set `permalink: /posts/YYYY/MM/slug/`.
- **Title comes from front matter.** Do NOT put `# Title` (H1) in the Markdown body — it double-renders.
- **URL changes need redirects.** Add `redirect_from:` to front matter listing old URL(s). Plugin `jekyll-redirect-from` handles the rest.
- **`README.md` is the homepage article index.** Add a reverse-chronological bullet for each new post.
- **`AGENTS.md` is excluded from Jekyll build** (see `_config.yml` `exclude`).

### Sidebar Tabs (`_tabs/`)

Chirpy uses `_tabs/` for sidebar navigation. Current tabs:
- `categories.md` (icon: `fa-stream`, order: 1)
- `tags.md` (icon: `fa-tags`, order: 2)
- `about.md` (icon: `fa-info-circle`, order: 4)

Add new tabs as Markdown files in `_tabs/` with `icon:`, `order:`, and optional `title:` front matter.

## Post conventions

Required front matter:
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

- **Images:** Two conventions coexist. Recent posts use `images/` (root dir) with paths like `/images/foo.png`. Older posts use `assets/images/foo.png`. Check the prevailing pattern before adding new images.
- Subtitle: `*Italic subtitle*` or `> Blockquote subtitle`
- Separators: `---` between major sections
- Math: `$$` block, `$` inline
- Code blocks: triple backticks with language tag

## Git

- `main` is production. Branch prefixes: `feature/`, `fix/`, `docs/`.
- Conventional commits preferred (`feat:`, `fix:`, `docs:`), not enforced.
- **No npm/Webpack/JS bundlers** — intentionally Jekyll-only.

## Troubleshooting

| Symptom | Fix |
| :--- | :--- |
| Liquid Exception | Check `{{ }}` / `{% %}` syntax in Markdown/SCSS |
| Stale content | `bundle exec jekyll clean` then rebuild |
| Old URL 404s | Add `redirect_from: /old-url` to post front matter |
| Wrong Ruby version | Use `export PATH="/opt/homebrew/opt/ruby@3.2/bin:$PATH"` before commands |
| Missing `jekyll-theme-chirpy` gem | Run `bundle install` with Homebrew Ruby 3.2 |
| Pagination broken | Ensure `index.html` exists at root with `layout: home` |
