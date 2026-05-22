---
permalink: /
title: "Semper ex Datis — Always from Data"
author_profile: true
---

**U.S. Marine | Data Scientist**

I write long-form technical articles about Large Language Models, Data Science, and machine learning — the kind that actually explain how things work under the hood.

---

### What You'll Find Here

- **Deep dives** into LLM architecture, fine-tuning, observability, and deployment
- **Data science** explorations — recommendation systems, computer vision, and people analytics
- **Practical guides** with code, math, and real-world implementation strategies
- **No fluff, no corporate gloss** — just technical depth and honest analysis

---

### Recent Writing

{% for post in site.posts limit:5 %}
- *{{ post.date | date: "%B %Y" }}* — [{{ post.title }}]({{ post.url }})
{% endfor %}

See all [blog posts](/year-archive/).
