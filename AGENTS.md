# AGENTS.md — gunnymarc.github.io

Personal blog/portfolio (Marc Buraczynski) — Jekyll static site, fork of academicpages/Minimal Mistakes, deployed via GitHub Pages on push to `main`.

## Commands

| Action | Command |
| :--- | :--- |
| Install deps | `bundle install` |
| Serve locally | `bundle exec jekyll serve --livereload` (http://localhost:4000) |
| Build | `bundle exec jekyll build` (output in `_site/`) |
| Clean cache | `bundle exec jekyll clean` |

**Ruby 3.1+ required** (system Ruby 2.6 is too old). No test suite — "testing" means `jekyll build` then inspect `_site/posts/YYYY/MM/slug/`.

## Architecture

- **Theme is forked, not gemmed.** `_layouts/`, `_includes/`, `_sass/` are in-repo and editable. Theme variants via `site_theme` in `_config.yml`: `default`, `air`, `sunrise`, `mint`, `dirt`, `contrast`.
- **CSS overrides** go in `assets/css/main.scss` **after** the `@import` block (earlier additions get clobbered).
- **Collections:** `_portfolio`, `_publications`, `_talks`, `_teaching` are Jekyll collections with `output: true` — content at `/portfolio/...`, `/talks/...`, etc.
- **Per-post `permalink:` is required in front matter.** Global `_config.yml` permalink is `/:categories/:title/` — do not rely on it. Always set `permalink: /posts/YYYY/MM/slug/`.
- **Title comes from front matter.** Do NOT put `# Title` (H1) in the Markdown body — it double-renders.
- **URL changes need redirects.** Add `redirect_from:` to front matter listing old URL(s). Plugin `jekyll-redirect-from` handles the rest.
- **`README.md` is the homepage article index.** Add a reverse-chronological bullet for each new post.
- **`AGENTS.md` is excluded from Jekyll build** (see `_config.yml` `exclude`).

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

- **Images:** Two conventions coexist. Recent posts use `images/` (root dir) with paths like `../images/foo.png` or `/images/foo.png`. Older posts use `assets/images/foo.png`. Check the prevailing pattern before adding new images.
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

## Related

- `CLAUDE.md` — sibling instruction file; delegates to this file as authoritative.
