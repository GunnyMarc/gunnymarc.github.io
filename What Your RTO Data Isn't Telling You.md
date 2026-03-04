---
---

# What Your RTO Data Isn't Telling You—And Why It Matters for Your People

Let's talk about something that's probably been nagging at you if you're in HR: those badge swipe reports everyone's using to justify return-to-office decisions? They might be telling you a story that isn't quite true.

Here's the thing—when we looked at how traditional analytics measure office attendance, we found they overestimate how much remote-preferring employees *actually* want to come in by about 33 percentage points. That's huge. Your dashboard might say someone's 45% office-inclined when they're really closer to 12%.

This isn't about blaming anyone or pointing fingers at your data team. It's about understanding a hidden bias that affects everyone—and more importantly, figuring out how to actually support your people based on what they truly need.

---

## Why This Happens: The Sarah and Marcus Problem

Let me paint a picture that probably sounds familiar.

**Sarah** is one of those people who genuinely loves the office. She lives nearby, enjoys the energy of being around colleagues, and comes in four days a week. Over six months, you've collected tons of data about her—which desk she prefers, what days work best, how she uses shared spaces.

**Marcus** is a senior engineer with a longer commute, two kids at home, and a really nice home office setup. He comes in maybe twice a month, usually for important meetings.

Here's the problem: you have *five times* more data about Sarah than Marcus. And that's not random—it's because Marcus simply isn't there to be observed.

This is what statisticians call **"Missing Not At Random" (MNAR)**[^1], and it creates a sneaky bias [Rubin, 1976; Little & Rubin, 2019]. The people who don't love the office... don't come to the office. When they don't come, we don't capture their preferences. So our data ends up dominated by people like Sarah.

When traditional tools try to fill in the gaps for Marcus, they use patterns from all that Sarah-heavy data—a technique known as **collaborative filtering** [Koren et al., 2009]. The result? They predict Marcus wants to come in way more than he actually does.

**From an HR perspective, this matters enormously.** If we're building policies, designing spaces, or making decisions based on data that doesn't accurately represent over half our workforce, we're setting ourselves up to create experiences that don't serve our people well.

[^1]: **Missing Not At Random (MNAR)** is a concept from statistical theory where the probability of data being missing depends on the unobserved value itself. In our context: people who dislike the office are less likely to come in, so their preferences remain unobserved. This differs from "Missing Completely At Random" (MCAR) or "Missing At Random" (MAR) scenarios where simpler imputation methods work well. For a comprehensive treatment, see Little & Rubin (2019).

---

## What We Actually Found When We Corrected for This Bias

We analyzed data from 500 employees over 26 weeks using a more sophisticated approach based on **Causal Matrix Completion** [Agarwal et al., 2023]. The findings were eye-opening—and honestly, they should change how we think about hybrid work.

### The Real Distribution of What People Want

| Who They Are | % of Your Team | What They Actually Prefer | What Standard Reports Say |
|:-------------|:---------------|:--------------------------|:--------------------------|
| **Remote-First Folks** | ~54% | Less than 30% office time | ~45% (way overestimated) |
| **Hybrid-Regulars** | ~23% | 30-60% office time | Roughly accurate |
| **Office Enthusiasts** | ~10% | 70%+ office time | Slightly underestimated |
| **Work-Life Balancers** | ~6% | It varies | Often misclassified |
| **Flexibility Seekers** | ~4% | Depends on context | Averaged incorrectly |
| **Collaboration Seekers** | ~3% | High when their team is there | Overgeneralized |

The headline? **Over half of your knowledge workers are remote-first.** They genuinely prefer being in the office less than 30% of the time. But standard analytics report them as wanting nearly half-and-half.

### Why This Happens: The Missing Data Gap

| Employee Type | How Much Data We Actually Capture | What's Missing |
|:--------------|:----------------------------------|:---------------|
| Office Enthusiasts | 82.5% | 17.5% |
| Middle-of-the-Road | 67.3% | 32.7% |
| Remote-Preferring | 45.2% | **54.8%** |

Remote-first employees have *three times* more missing data than office enthusiasts. Every decision we make using this data inherits that imbalance—unless we correct for it. This observation pattern is consistent with research on **selection bias in observational data** [Heckman, 1979; Imbens & Rubin, 2015].

---

## A Better Way to Understand Your People: Causal Matrix Completion

There's a technique called **Causal Matrix Completion** [Agarwal, Dahleh, Shah, & Shen, 2023] that helps us see through this noise. I won't bore you with all the math, but here's the intuition:

Instead of filling in missing data using patterns from people who *are* observed (which skews toward office enthusiasts), this approach asks: "What would we see if everyone had an equal chance of showing up?"[^2]

It works by giving more weight to rare observations—a technique called **Inverse Propensity Weighting (IPW)** [Rosenbaum & Rubin, 1983]. If someone who almost never comes in *does* show up, that tells us something meaningful—so we weight that information more heavily.

[^2]: This counterfactual framing draws on the **potential outcomes framework** developed by Rubin (1974) and expanded in the causal inference literature. The key question is: "What would this person's observed office behavior look like under different conditions?" See Imbens & Rubin (2015) for a comprehensive treatment of this framework.

### How Causal Matrix Completion Works

The technical foundation combines insights from three fields:

1. **Matrix Completion** [Candès & Recht, 2009]: The idea that we can recover missing entries in a matrix if the underlying data has low-rank structure (i.e., people's preferences cluster into a few underlying types).

2. **Propensity Score Methods** [Rosenbaum & Rubin, 1983]: Weighting observations by the inverse probability of being observed to correct for selection bias.

3. **Synthetic Control Methods** [Abadie et al., 2010]: Using similar individuals to estimate counterfactual outcomes.

Agarwal et al. (2023) unified these approaches for panel data settings where both the outcome and the treatment/observation probability can be correlated with unobserved factors—exactly our office attendance scenario.[^3]

[^3]: The mathematical guarantee requires what researchers call a **latent factor model**: the assumption that employee preferences and observation patterns are driven by a small number of underlying factors. In workplace contexts, these factors might include commute tolerance, collaboration needs, and home office quality. See Section 3 of Agarwal et al. (2023) for the formal conditions.

### The Difference It Makes

| What We're Measuring | Traditional Approach | Corrected Approach | Improvement |
|:---------------------|:---------------------|:-------------------|:------------|
| Accuracy for remote-first employees | Pretty poor | Much better | 43% improvement |
| Bias for remote-first employees | Overestimates by 33 points | Overestimates by 8 points | 75% reduction |

**What this means in practical terms:** Traditional analytics might tell you to plan for 45% office attendance from your remote-first folks. The corrected approach predicts closer to 20%. That's the difference between designing for 60% occupancy and 40% occupancy—which has real implications for real estate costs, space design, and how your people experience the workplace.

---

## The Four Things That Actually Drive Office Behavior

Once we could measure preferences accurately, clear patterns emerged. Understanding these can help you support different employees more effectively. This analysis builds on research in **behavioral propensity modeling** [Ajzen, 1991; Davis, 1989].

### 1. Their Baseline Preference (Office vs. Remote)

This is someone's natural tendency—do they gravitate toward the office or toward working from home? In statistical terms, this is their **propensity for office attendance** [Austin, 2011].

**What influences it most:**
- **Commute distance** is the biggest factor [Bloom et al., 2015]. Every additional kilometer of commute reduces office preference. A 30km commute basically cancels out any baseline preference for the office.
- **Role matters.** Sales folks and managers tend to prefer the office more; engineers and data scientists tend to prefer remote [Choudhury et al., 2021].
- **Home office quality** makes a 29 percentage point difference between people with poor setups and those with great ones.
- **Having kids at home** reduces office preference (flexibility needs are real) [Barrero et al., 2021].

**One surprise:** Age wasn't a significant factor. Contrary to stereotypes, millennials aren't necessarily more remote-preferring than Gen X or Boomers once you account for other things [Bloom, 2020].

### 2. How Much They Value In-Person Collaboration

Some people genuinely need face-to-face interaction to do their best work. Others thrive with async tools like Slack or email [Yang et al., 2022].

Here's something interesting: true collaboration-seekers make up only about 3% of the workforce, but they're force multipliers. When they're in the office, others tend to follow—a phenomenon related to **social influence in networks** [Aral & Walker, 2012]. Consider them your "anchor" employees for team sync days.

### 3. What They Think About Office Amenities

Here's a counterintuitive finding: **amenities are retention tools, not attraction tools.**

That fancy cafeteria and new gym? They keep your office enthusiasts happy, but they won't convince someone with a 90-minute round-trip commute to come in more often. Free lunch is lovely, but it doesn't offset a long, stressful commute or competing caregiving responsibilities. This aligns with research on **workplace satisfaction factors** [Herzberg, 1959; Judge et al., 2001].

Don't invest in amenities expecting them to increase attendance. Invest in them to support the people who are already choosing to be there.

### 4. How Much Flexibility Matters to Them

This one has an almost perfect inverse relationship with office preference [Barrero et al., 2021]. People who highly value flexibility almost by definition avoid fixed office schedules.

**HR heads-up:** Your flexibility-seekers (about 4% of the workforce) are the hardest to plan around. They won't commit to regular schedules—and forcing them to increases turnover risk [Mas & Pallais, 2017]. These are often your high performers who've earned that autonomy. Think carefully before creating one-size-fits-all policies.

---

## The Feedback Loop We Need to Talk About

Here's something that gets philosophically interesting—and has real ethical implications for HR.

Office recommendation systems don't just *predict* behavior. They *change* it. This phenomenon is well-documented in the **recommender systems literature** [Adomavicius & Tuzhilin, 2005; Ricci et al., 2015].

### How This Works

**Exposure Effect:** Someone who never comes on Mondays might assume the office is chaotic that day. If we recommend they try Monday (knowing it's actually quiet and productive), they might discover they like it [Zajonc, 1968]. Now their Monday preference increases, and future Monday recommendations are more likely accepted.

**Social Influence:** Recommend the same day to an entire team, everyone shows up, collaboration happens, everyone's preference for that day increases [Cialdini & Goldstein, 2004]. The effect compounds.

**Habit Formation:** Behavioral research tells us habits form after about 66 days of repetition [Lally et al., 2010]. If we nudge someone into a pattern for two months, that pattern tends to stick.

**Norm Formation:** Aggregate enough individual nudges and you've created culture. "Tuesday-Wednesday are office days" can emerge not from policy but from reinforced recommendations. New hires adopt it automatically—a process studied in **organizational culture research** [Schein, 2010].

### The Questions We Should Be Asking Ourselves

This creates some uncomfortable territory related to the ethics of **algorithmic nudging** [Thaler & Sunstein, 2008; Yeung, 2017]:
- If someone's preference gradually changes through recommendation-driven exposure, did we help them discover something about themselves? Or did we manipulate them?
- At what point does a helpful recommendation system become a subtle compliance mechanism?

There's no perfect answer here. But as HR professionals, we have a responsibility to consider:
- **Transparency:** Do people understand how recommendations are generated? [Diakopoulos, 2016]
- **Control:** Can they opt out?
- **Whose interests are we serving?** Do recommendations help employees, or just make scheduling easier for management?

---

## What You Can Do Tomorrow Morning

### For HR Leaders and People Ops

**1. Take badge data with a grain of salt.** It's systematically biased toward people who already like the office [Agarwal et al., 2023]. At minimum, look at different employee segments separately instead of treating everyone as one group.

**2. Build flexibility into your policies.** A blanket "3 days per week" mandate will get maybe 30% compliance from your remote-first majority—and probably some resentment along with it [Barrero et al., 2021]. Consider:
- 1 day/week for remote-first folks (critical meetings and collaboration)
- 2-3 days for hybrid workers
- Whatever works for your office enthusiasts

**3. Watch trends, not just attendance numbers.** If overall office preference is declining, find out why. Is the office experience getting worse? Are people setting up better home offices? Is manager pressure backfiring?

**4. Have honest conversations.** Survey your people. Ask them what they actually need. The data can tell you patterns, but only your people can tell you *why*.

### For Facilities and Real Estate Decisions

**1. Plan for 40-50% peak attendance, not 100%.** Our data shows about 37% average attendance with significant day-to-day variation (Tuesday-Wednesday peaks, Monday-Friday valleys).

**2. Design for different needs.** Not everyone uses the office the same way. A rough breakdown:
- ~15% permanent desks (for your office enthusiasts)
- ~35% hoteling/flexible workstations (for hybrid workers)
- ~25% meeting and collaboration spaces
- ~15% quiet pods for focused work
- ~10% amenities and social spaces

**3. Watch for crowding problems.** If everyone's being pushed toward Tuesday-Wednesday, you'll create miserable peak days and ghost-town off-days. That's bad for everyone's experience.

### For Managers Leading Hybrid Teams

**1. Coordinate, don't mandate.** "Tuesday is our team sync day—consider joining if it works for you" respects individual circumstances while signaling when in-person connection is most valuable.

**2. Protect your remote-first people from proximity bias.** Make sure they're evaluated on what they produce, not how often you see them [Choudhury et al., 2021]. Document decisions; don't rely on hallway conversations they can't be part of.

**3. Default to async.** Design meetings to accommodate remote participants as first-class citizens. And honestly, if it can be an email or a Slack message, make it one. Your people will thank you.

---

## The Bigger Picture

The hybrid work conversation has been full of strong opinions and not enough good data. Executives point to badge reports showing 40% attendance and conclude people want to return. Employees feel like mandates are disconnected from reality.

Both sides are working with incomplete information. The data has been biased—but it's fixable.

Getting this right isn't just about technical accuracy. It's about **honesty**. It's about acknowledging that most knowledge workers, when given genuine choice, prefer more flexibility than we've traditionally offered [Bloom et al., 2015; Barrero et al., 2021]. That's not a cultural failure or a problem to solve. It's information to incorporate into how we support our people.

Organizations that build policies on accurate data will make better decisions, see lower turnover, and create more nuanced approaches to collaboration. Those that don't will keep investing in things that don't move the needle and creating mandates that don't improve engagement.

**The data exists. The methods exist. The real question is: are we ready to listen to what it's actually saying—and to design workplaces that genuinely serve the people in them?**

---

## A Quick Note on the Technical Stuff

If you're curious about the methodology behind these findings, the approach is called **Causal Matrix Completion with Inverse Propensity Weighting** [Agarwal et al., 2023]. In plain English: it corrects for the fact that we have more data about some people than others, and weights observations accordingly.

The key insight is simple: when data is "missing not at random"—meaning the people we don't observe are systematically different from those we do [Little & Rubin, 2019]—traditional analytics will mislead us. Causal approaches help us see a more complete picture.

You don't need to become a data scientist to apply these insights. But if you're working with analytics teams on workforce planning, sharing this perspective might spark some valuable conversations about how to get better, more equitable insights from your data.

---

*The goal isn't to prove that remote work is "better" or office work is "wrong." It's to understand what your people actually need—and to build policies that support them in doing their best work, wherever that happens.*

---

## References

### Causal Matrix Completion & Missing Data

Agarwal, A., Dahleh, M., Shah, D., & Shen, D. (2023). Causal matrix completion. *Proceedings of The 34th International Conference on Algorithmic Learning Theory (ALT 2023)*, PMLR 195, 3-36. https://proceedings.mlr.press/v195/agarwal23c

Agarwal, A., Dahleh, M., Shah, D., & Shen, D. (2021). Causal matrix completion. *arXiv preprint arXiv:2109.15154*. https://arxiv.org/abs/2109.15154

Little, R. J. A., & Rubin, D. B. (2019). *Statistical analysis with missing data* (3rd ed.). Wiley.

Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581-592.

### Propensity Scores & Causal Inference

Austin, P. C. (2011). An introduction to propensity score methods for reducing the effects of confounding in observational studies. *Multivariate Behavioral Research*, 46(3), 399-424.

Heckman, J. J. (1979). Sample selection bias as a specification error. *Econometrica*, 47(1), 153-161.

Imbens, G. W., & Rubin, D. B. (2015). *Causal inference for statistics, social, and biomedical sciences: An introduction*. Cambridge University Press.

Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55.

Rubin, D. B. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies. *Journal of Educational Psychology*, 66(5), 688-701.

### Matrix Completion & Recommender Systems

Adomavicius, G., & Tuzhilin, A. (2005). Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions. *IEEE Transactions on Knowledge and Data Engineering*, 17(6), 734-749.

Candès, E. J., & Recht, B. (2009). Exact matrix completion via convex optimization. *Foundations of Computational Mathematics*, 9(6), 717-772.

Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30-37.

Ricci, F., Rokach, L., & Shapira, B. (Eds.). (2015). *Recommender systems handbook* (2nd ed.). Springer.

### Synthetic Controls & Panel Data

Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies: Estimating the effect of California's tobacco control program. *Journal of the American Statistical Association*, 105(490), 493-505.

Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021). Matrix completion methods for causal panel data models. *Journal of the American Statistical Association*, 116(536), 1716-1730.

### Remote Work & Workplace Research

Barrero, J. M., Bloom, N., & Davis, S. J. (2021). Why working from home will stick. *NBER Working Paper No. 28731*. https://www.nber.org/papers/w28731

Bloom, N. (2020). How working from home works out. *Stanford Institute for Economic Policy Research (SIEPR) Policy Brief*. https://siepr.stanford.edu/publications/policy-brief/how-working-home-works-out

Bloom, N., Liang, J., Roberts, J., & Ying, Z. J. (2015). Does working from home work? Evidence from a Chinese experiment. *The Quarterly Journal of Economics*, 130(1), 165-218.

Choudhury, P., Foroughi, C., & Larson, B. (2021). Work‐from‐anywhere: The productivity effects of geographic flexibility. *Strategic Management Journal*, 42(4), 655-683.

Mas, A., & Pallais, A. (2017). Valuing alternative work arrangements. *American Economic Review*, 107(12), 3722-3759.

Yang, L., Holtz, D., Jaffe, S., Suri, S., Sinha, S., Weston, J., ... & Teevan, J. (2022). The effects of remote work on collaboration among information workers. *Nature Human Behaviour*, 6(1), 43-54.

### Behavioral Science & Organizational Psychology

Ajzen, I. (1991). The theory of planned behavior. *Organizational Behavior and Human Decision Processes*, 50(2), 179-211.

Aral, S., & Walker, D. (2012). Identifying influential and susceptible members of social networks. *Science*, 337(6092), 337-341.

Cialdini, R. B., & Goldstein, N. J. (2004). Social influence: Compliance and conformity. *Annual Review of Psychology*, 55, 591-621.

Davis, F. D. (1989). Perceived usefulness, perceived ease of use, and user acceptance of information technology. *MIS Quarterly*, 13(3), 319-340.

Herzberg, F. (1959). *The motivation to work*. Wiley.

Judge, T. A., Thoresen, C. J., Bono, J. E., & Patton, G. K. (2001). The job satisfaction–job performance relationship: A qualitative and quantitative review. *Psychological Bulletin*, 127(3), 376-407.

Lally, P., Van Jaarsveld, C. H., Potts, H. W., & Wardle, J. (2010). How are habits formed: Modelling habit formation in the real world. *European Journal of Social Psychology*, 40(6), 998-1009.

Schein, E. H. (2010). *Organizational culture and leadership* (4th ed.). Jossey-Bass.

Zajonc, R. B. (1968). Attitudinal effects of mere exposure. *Journal of Personality and Social Psychology*, 9(2, Pt. 2), 1-27.

### Ethics of Algorithms & Nudging

Diakopoulos, N. (2016). Accountability in algorithmic decision making. *Communications of the ACM*, 59(2), 56-62.

Thaler, R. H., & Sunstein, C. R. (2008). *Nudge: Improving decisions about health, wealth, and happiness*. Yale University Press.

Yeung, K. (2017). 'Hypernudge': Big Data as a mode of regulation by design. *Information, Communication & Society*, 20(1), 118-136.

---

## Further Reading

### For HR Professionals New to These Concepts

- **On missing data basics:** Schafer, J. L., & Graham, J. W. (2002). Missing data: Our view of the state of the art. *Psychological Methods*, 7(2), 147-177.

- **On propensity scores (accessible intro):** Caliendo, M., & Kopeinig, S. (2008). Some practical guidance for the implementation of propensity score matching. *Journal of Economic Surveys*, 22(1), 31-72.

- **On the future of work:** Gratton, L. (2022). *Redesigning Work: How to Transform Your Organization and Make Hybrid Work for Everyone*. MIT Press.

### For Data Scientists & Analytics Teams

- **Foundational matrix completion:** Candès, E. J., & Plan, Y. (2010). Matrix completion with noise. *Proceedings of the IEEE*, 98(6), 925-936.

- **Causal inference with interference:** Athey, S., Eckles, D., & Imbens, G. W. (2018). Exact p-values for network interference. *Journal of the American Statistical Association*, 113(521), 230-240.

- **Deep learning for recommenders:** Zhang, S., Yao, L., Sun, A., & Tay, Y. (2019). Deep learning based recommender system: A survey and new perspectives. *ACM Computing Surveys*, 52(1), 1-38.

### Key Online Resources

- **Devavrat Shah's research page:** https://devavrat.mit.edu/ — Primary author on Causal Matrix Completion with extensive publications on machine learning and causal inference.

- **NBER Working Papers on Remote Work:** https://www.nber.org/topics/labor-studies — Ongoing research from Bloom, Barrero, and Davis on work-from-home economics.

- **Causal Inference: The Mixtape (free online book):** https://mixtape.scunning.com/ — Accessible introduction to causal inference methods by Scott Cunningham.

- **Missing Data Wiki:** https://stefvanbuuren.name/fimd/ — Companion site to van Buuren's "Flexible Imputation of Missing Data" textbook.

- **arXiv Causal Matrix Completion paper:** https://arxiv.org/abs/2109.15154 — Full technical paper with proofs and extended analysis.

- **ALT 2023 Proceedings version:** https://proceedings.mlr.press/v195/agarwal23c — Peer-reviewed conference publication.

---

*Document Version: With References | Last Updated: February 2026*
