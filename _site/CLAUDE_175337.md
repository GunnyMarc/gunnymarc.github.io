# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Authoritative Agent Guide

**`AGENTS.md` is the single source of truth** for this repository's conventions (build commands, front matter format, file layout, troubleshooting). Read it before making any non-trivial change. The notes below are a quick orientation — when this file and `AGENTS.md` disagree, `AGENTS.md` wins.

## What this repo is

Personal blog / portfolio for Marc Buraczynski at [gunnymarc.github.io](https://gunnymarc.github.io). Long-form technical articles on LLMs, data science, and ML. Jekyll static site, fork of the [academicpages](https://github.com/academicpages/academicpages.github.io) theme (Minimal Mistakes lineage). Deployed via GitHub Pages on push to `main`.

## Common commands

Requires Ruby 3.1+ (system Ruby 2.6 is too old — `brew install ruby@3.1` and add it to `PATH`).

```bash
bundle install                            # install deps
bundle exec jekyll serve --livereload     # serve at http://localhost:4000
bundle exec jekyll build                  # build into _site/
bundle exec jekyll clean                  # clear cache when content goes stale
```

There is no test suite. "Testing an article" means building and inspecting `_site/posts/YYYY/MM/slug/`.

## Architecture notes that aren't obvious from the tree

- **Theme is forked, not gemmed.** `_layouts/`, `_includes/`, and `_sass/` are part of the repo and editable. Theme variants are toggled via `site_theme` in `_config.yml` (`default`, `air`, `sunrise`, `mint`, `dirt`, `contrast`).
- **CSS overrides go in `assets/css/main.scss` _after_ the `@import` block** — earlier overrides get clobbered by the theme import.
- **Posts use a custom permalink scheme**: `/posts/YYYY/MM/slug/`, set per-post via `permalink:` in front matter (not derived from the filename's `_config.yml` `permalink:` setting, which is `/:categories/:title/`). Always set `permalink` explicitly on new posts.
- **Collections beyond `_posts`**: `_portfolio`, `_publications`, `_talks`, `_teaching` are configured as Jekyll collections with `output: true` and their own permalink schemes — content lands at `/portfolio/...`, `/talks/...`, etc.
- **Title is rendered from front matter**, not Markdown. Do NOT put `# Title` as an H1 in the post body — it will double-render.
- **URL changes require redirects.** If a post's `permalink` changes, add `redirect_from:` to the front matter listing the previous URL(s). The `jekyll-redirect-from` plugin handles the rest.
- **`README.md` is also the homepage article index.** When adding a new post, add a bullet in reverse-chronological order to the Articles list in `README.md`.

## Conventions worth surfacing up front

- Image paths: `assets/images/...` (not `images/...`).
- Branching: `main` is production. Use `feature/`, `fix/`, `docs/` prefixes.
- Conventional commits (`feat:`, `fix:`, `docs:`, …).
- Do not introduce npm / Webpack / JS bundlers — the toolchain is intentionally Jekyll-only.
- Match the existing articles' tone and mathematical rigor when writing or editing prose; read a recent post in `_posts/` first.
- Per `AGENTS.md`: do not summarize file-write actions unless explicitly asked.
