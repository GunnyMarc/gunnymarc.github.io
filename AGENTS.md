# AGENTS.md — gunnymarc.github.io

Personal blog/portfolio (Marc Buraczynski) — Jekyll static site, fork of academicpages/Minimal Mistakes, deployed via GitHub Pages on push to `main`.

## Stack & Commands

| Action | Command |
| :--- | :--- |
| Install deps | `bundle install` |
| Serve locally | `bundle exec jekyll serve --livereload` (http://localhost:4000) |
| Build | `bundle exec jekyll build` (output in `_site/`) |
| Clean cache | `bundle exec jekyll clean` |

**Ruby:** 3.1+ required (system Ruby 2.6 is too old). `brew install ruby@3.1` then add to `PATH`.

**No test suite.** "Testing" means `bundle exec jekyll build` then inspect `_site/posts/YYYY/MM/slug/`.

## Architecture

- **Theme is forked, not gemmed.** `_layouts/`, `_includes/`, `_sass/` are in-repo and editable. Theme variants set via `site_theme` in `_config.yml`: `default`, `air`, `sunrise`, `mint`, `dirt`, `contrast`.
- **CSS overrides** go in `assets/css/main.scss` **after** the `@import` block (earlier additions get clobbered).
- **Collections beyond `_posts`:** `_portfolio`, `_publications`, `_talks`, `_teaching` are Jekyll collections with `output: true` — content lands at `/portfolio/...`, `/talks/...`, etc.
- **Posts use explicit per-post `permalink:` in front matter.** The global `_config.yml` permalink is `/:categories/:title/` — do not rely on it. Always set `permalink: /posts/YYYY/MM/slug/` on each new post.
- **Title is rendered from front matter.** Do NOT put `# Title` (H1) in the Markdown body — it will double-render.
- **URL changes require redirects.** Add `redirect_from:` to front matter when changing a post's `permalink`. Plugin `jekyll-redirect-from` handles the rest.
- **`README.md` is the homepage article index.** When adding a post, insert a bullet in reverse-chronological order: `- *Month Year* — [Title](https://gunnymarc.github.io/posts/YYYY/MM/slug/)`

## Article Conventions

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
- Images path: `assets/images/...` (not `images/...`). Used relative in Markdown: `![alt](assets/images/file.png)`
- Subtitle: optional `*Italic subtitle*` or `> Blockquote subtitle`
- Separators: `---` between major sections
- Math: `$$` block, `$` inline
- Code blocks: triple backticks with language tag

## Git

- `main` is production. Branch prefixes: `feature/`, `fix/`, `docs/`.
- Conventional commits preferred (`feat:`, `fix:`, `docs:`, …) but not enforced.
- Do not commit secrets or introduce npm/Webpack/JS bundlers.

## Troubleshooting

| Symptom | Fix |
| :--- | :--- |
| Liquid Exception | Check `{{ }}` / `{% %}` syntax in Markdown/SCSS |
| Stale content | `bundle exec jekyll clean` then rebuild |
| Old URL 404s | Add `redirect_from: /old-url` to post front matter |
