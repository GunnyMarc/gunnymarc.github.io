# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Site Overview

This is a GitHub Pages site ([gunnymarc.github.io](https://gunnymarc.github.io)) powered by Jekyll using the `jekyll-theme-midnight` theme. It's a personal blog focused on data science, AI, and technology topics by a U.S. Marine / Data Scientist.

## Architecture

- **`_config.yml`** — Jekyll site configuration (theme, title, description)
- **`README.md`** — Serves as the site homepage/index; contains the article listing and site disclaimer
- **`*.md` files** (excluding README.md and CLAUDE.md) — Individual blog post articles

There is no build toolchain in this repo. GitHub Pages builds and deploys the site automatically on push to `main`.

## Content Structure

New articles are added as Markdown files in the root directory, then linked from `README.md` under the `## Articles` section. Article URLs follow the pattern `https://gunnymarc.github.io/<filename-without-.md>`.

## Local Preview

To preview locally, install Jekyll and run:

```bash
gem install jekyll bundler
jekyll serve
```

The site will be available at `http://localhost:4000`.
