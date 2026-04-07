\### Slide 1: Title Slide



\*\*Main Title (very large, bold, centered):\*\*  

\*\*Mamba → Mamba-3\*\*  

\*\*Evolution of Selective State Space Models for Efficient Generative AI\*\*



\*\*Subtitle (medium size, centered):\*\*  

Research Paper Study



\*\*Bottom Section (small, clean text):\*\*  

\[Your Group Name / Group 5]  

Member 1 Member 2 Member 3 Member 4 Member 5  

\[Course Name]  April 2026



\*\*Visual Suggestion (highly recommended):\*\*

\- Very clean background with subtle blue/purple gradient  

\- Optional small icon: A simple stylized “M” or waveform symbol representing sequence modeling  

\- Lots of white space — keep the slide elegant and uncluttered



\### Why This Works

\- Extremely low text — only the essential information  

\- Professional first impression  

\- Clearly states the topic and shows the evolution from Mamba to Mamba-3  

\- Audience immediately understands what the presentation is about



\### Suggested Speaker Notes (what you say):

"Good \[morning/afternoon], everyone. Today our group will present on the evolution of Selective State Space Models — from the original Mamba paper in 2023 to the recent improvements in Mamba-3. This work represents an important direction in making generative AI more efficient."



\### Slide 2: Agenda



\*\*Slide Title (large, bold, centered at top):\*\*  

\*\*Agenda\*\*



\*\*Main Content (centered, large font, lots of white space):\*\*



\- The Problem \& Motivation  

\- State Space Models – Quick Background  

\- Original Mamba: Core Innovations  

\- Experiments \& Key Results  

\- Limitations of Mamba  

\- Introducing Mamba-3 (2026)  

\- Mamba-3 Innovations \& Improvements  

\- Significance, Limitations \& Future Directions  



\*\*Small subtle line at the bottom (smallest font):\*\*  

Efficient sequence modeling for Generative AI



\*\*Visual Suggestion (recommended):\*\*  

\- A clean vertical numbered list or simple flowchart/arrow going downward on the right side of the slide.  

\- Use subtle icons (e.g., ⚡ for problem, 🔬 for methodology, 📊 for results, 🔄 for evolution).  

\- Keep background minimal with your chosen blue/purple academic theme.



\### Why This Version is Good

\- Extremely low text — only short phrases.  

\- Easy for the audience to scan in seconds.  

\- Clearly communicates the structure without overwhelming anyone.  

\- Leaves most of the talking to you (ideal for a 15–20 minute presentation).



\### Suggested Speaker Notes (what you say):

"Good morning everyone. Today’s presentation follows this agenda. We’ll begin with the core problem that led to the development of Mamba, build some intuition around state space models, explore the original Mamba paper in detail, look at its results and limitations, and then examine how Mamba-3 improves upon it in 2026. This evolution highlights important advances in making generative AI more efficient."



\### Slide 3: The Core Problem



\*\*Slide Title (large, bold):\*\*  

\*\*The Core Problem\*\*



\*\*Main Content (large font, lots of white space, centered or left-aligned with bullets):\*\*



\- Transformers scale quadratically  

\- O(n²) time \& memory with sequence length  

\- KV-cache explodes for long contexts  

\- Slow \& expensive inference  

\- Prior SSMs too rigid (non-selective)



\*\*Small subtle line at bottom (very small font):\*\*  

Why Mamba was needed (Gu \& Dao, 2023)



\*\*Visual Suggestion (strongly recommended – this is what makes the slide effective):\*\*

\- Left side: A simple exploding quadratic curve graph (O(n²) line shooting upward)

\- Right side: Icon of a slow/heavy Transformer model vs. a bottleneck symbol

\- Keep the slide mostly visual — text should be minimal and large



\### Why This Works

\- Extremely low text (only short phrases)

\- Audience can read everything in <10 seconds

\- Focus stays on your explanation

\- Clearly defines the problem as required by the assignment



\### Suggested Speaker Notes (what you actually say):

"The main problem addressed in the Mamba paper is the quadratic scaling of Transformers. As sequence length increases, both computation time and memory usage grow with the square of n. This makes long-context generation extremely slow and memory-hungry due to the growing KV-cache. Earlier State Space Models were efficient in theory but too rigid — they couldn’t selectively focus on important information. This fundamental limitation motivated the development of Mamba in 2023."



\### Slide 4: Motivation



\*\*Slide Title (large, bold):\*\*  

\*\*Motivation\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Need linear-time sequence models  

\- Must handle long-range dependencies  

\- Require input-dependent selectivity  

\- Keep constant memory \& high throughput  

\- For real-world generative AI



\*\*Small subtle line at bottom (very small font):\*\*  

Gu \& Dao (2023)



\*\*Visual Suggestion (highly recommended):\*\*

\- Simple icon or diagram on the right:  

&nbsp; - A “smart filter” or gate icon (representing selectivity)  

&nbsp; - Arrow from rigid SSM → Selective Mamba  

&nbsp; - Or a speedometer showing “Fast + Efficient”  

\- Keep the slide 70% visual / 30% text



\### Why This Works

\- Extremely short bullets (easy to read in seconds)  

\- Focus stays on your spoken explanation  

\- Clearly covers the assignment requirement (motivation \& limitations of existing approaches)  

\- Flows naturally from Slide 3 (Problem)



\### Suggested Speaker Notes (what you say):

"The motivation behind Mamba was simple but powerful. We needed sequence models that scale linearly with length, can remember important long-range information, and dynamically select what to keep or forget based on the input. Previous models either scaled poorly or lacked this selectivity while keeping memory constant. Mamba was designed to solve exactly this gap for modern generative AI applications."



\### Slide 5: State Space Models – Background



\*\*Slide Title (large, bold):\*\*  

\*\*State Space Models\*\*  

\*\*Background\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- SSM = continuous system discretized  

\- Hidden state compresses entire history  

\- HiPPO \& S4 laid the foundation  

\- Linear time \& constant memory  

\- But lacked selectivity



\*\*Small subtle line at bottom (very small font):\*\*  

Intuitive overview (Gu \& Dao, 2023)



\*\*Visual Suggestion (highly recommended):\*\*

\- Center or right side: A simple diagram showing  

&nbsp; - Input → Hidden State (recurrence) → Output  

&nbsp; - Or a timeline with a “memory bottle” compressing past tokens  

\- One clean SSM recurrence equation in KaTeX (very large):  

&nbsp; \\( h\_t = A h\_{t-1} + B x\_t \\)  

\- Keep the slide mostly visual — text should feel light.



\### Why This Works

\- Extremely low text (only short phrases)  

\- Audience gets the big picture instantly  

\- Builds intuition without overwhelming details  

\- Perfect bridge from motivation to Mamba’s innovations



\### Suggested Speaker Notes (what you say):

"Before Mamba, let’s quickly understand State Space Models. An SSM treats the sequence as a continuous dynamical system that gets discretized for neural networks. It maintains a hidden state that compresses all previous information into a fixed-size vector. Earlier work like HiPPO and S4 showed this could run in linear time with constant memory — a huge theoretical advantage over Transformers. However, they were still rigid and lacked the ability to selectively focus on important information. This is exactly where Mamba comes in."



Slide 6: Original Mamba – Key Innovations (Part 1)



\*\*Slide Title (large, bold):\*\*  

\*\*Original Mamba\*\*  

\*\*Key Innovations (Part 1)\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Selective State Space (S6)  

\- Input-dependent parameters  

\- A, B, C matrices change per token  

\- Enables dynamic selection  

\- Focus or forget relevant info



\*\*Small subtle line at bottom (very small font):\*\*  

Gu \& Dao (2023)



\*\*Visual Suggestion (highly recommended):\*\*

\- Center/right side: Clean diagram of the selective mechanism  

&nbsp; - Show three matrices (A, B, C) with arrows changing based on input token  

&nbsp; - Or a simple “gate” icon: Input → Selective Filter → Updated State  

\- One large KaTeX equation (keep it very clean):  

&nbsp; \\( \\Delta\_t, B\_t, C\_t = \\text{select}(x\_t) \\)



\### Why This Works

\- Extremely low text — only short phrases  

\- Audience can read it instantly  

\- Focus stays on your spoken explanation  

\- Clearly highlights the core innovation from the Mamba paper



\### Suggested Speaker Notes (what you say):

"The first major innovation in the original Mamba paper is the Selective State Space model, or S6. Unlike previous SSMs where the parameters were fixed, Mamba makes the A, B, and C matrices input-dependent. For every new token, the model dynamically decides how to update its hidden state. This selectivity allows the model to focus on important information and forget irrelevant details — something rigid SSMs couldn’t do. This is the key idea that makes Mamba powerful for real data like language."



Slide 7: Original Mamba – Key Innovations (Part 2)



\*\*Slide Title (large, bold):\*\*  

\*\*Original Mamba\*\*  

\*\*Key Innovations (Part 2)\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Hardware-aware parallel scan  

\- Computes recurrence efficiently  

\- Simplified Mamba block  

\- No attention mechanism  

\- Minimal MLPs → linear scaling



\*\*Small subtle line at bottom (very small font):\*\*  

Gu \& Dao (2023)



\*\*Visual Suggestion (highly recommended):\*\*

\- Center or right side: Clean architecture diagram of the full Mamba block  

&nbsp; - Show input → Selective SSM → Gating → Output  

&nbsp; - Small arrow labeled “Parallel Scan”  

&nbsp; - Or a side-by-side comparison: Transformer (complex) vs. Mamba block (simple)  

\- Keep the diagram large and uncluttered



\### Why This Works

\- Extremely low text — only short phrases  

\- Audience reads it in seconds  

\- Focus stays on your explanation  

\- Completes the methodology section without overload



\### Suggested Speaker Notes (what you say):

"The second major innovation is the hardware-aware parallel scan. This algorithm lets the model compute the entire sequence recurrence in parallel, making training and inference extremely fast. Combined with a greatly simplified block architecture — no attention heads and almost no MLPs — Mamba achieves true linear scaling in both time and memory. This is what allows it to run 5× faster than Transformers while keeping the model simple."



\### Slide 8: Experimental Setup



\*\*Slide Title (large, bold):\*\*  

\*\*Experimental Setup\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Datasets: The Pile, Long Range Arena  

\- Metrics: Perplexity, accuracy, throughput  

\- Memory \& scaling tests  

\- Compared with same-size Transformers  



\*\*Small subtle line at bottom (very small font):\*\*  

Gu \& Dao (2023)



\*\*Visual Suggestion (highly recommended):\*\*

\- Right side: Simple table or icons  

&nbsp; - Book icon → The Pile  

&nbsp; - Target icon → Long Range Arena  

&nbsp; - Graph icon → Scaling curves  

\- Or a clean 2-column layout: “What we tested” vs. “How we measured”



\### Why This Works

\- Extremely low text — only short phrases  

\- Easy to scan in seconds  

\- Focus stays on your spoken explanation  

\- Covers the assignment requirement (experimental setup \& datasets) without clutter



\### Suggested Speaker Notes (what you say):

"For evaluation, the authors used large-scale language modeling on The Pile dataset and the Long Range Arena benchmark for long-context tasks. They measured perplexity, downstream accuracy, inference speed, memory usage, and how well the model scales with sequence length. All experiments compared Mamba directly against same-size Transformer baselines."



\### Slide 9: Results of Original Mamba



\*\*Slide Title (large, bold):\*\*  

\*\*Results of Original Mamba\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Matches / exceeds same-size Transformers  

\- 5× higher inference throughput  

\- Linear scaling with sequence length  

\- Strong performance up to 1M+ tokens  

\- Mamba-3B matches Transformers 2× its size



\*\*Small subtle line at bottom (very small font):\*\*  

Gu \& Dao (2023)



\*\*Visual Suggestion (highly recommended – this slide benefits greatly from visuals):\*\*

\- Right or center: A bar chart or line graph showing  

&nbsp; - Perplexity comparison (Mamba vs Transformer)  

&nbsp; - Throughput bar: Mamba 5× higher  

&nbsp; - Scaling curve: Linear (Mamba) vs Quadratic (Transformer)  

\- Or a simple “5× Faster” icon with an arrow



\### Why This Works

\- Extremely low text — short, scannable phrases only  

\- Audience can grasp the key wins instantly  

\- Focus stays on your spoken explanation  

\- Highlights the most impactful results from the paper without clutter



\### Suggested Speaker Notes (what you say):

"The original Mamba achieved impressive results. On language modeling, it matches or exceeds Transformers of the same size in both pretraining perplexity and downstream tasks. Most notably, it delivers up to 5 times higher inference throughput while scaling linearly with sequence length instead of quadratically. The authors showed strong performance even on sequences up to one million tokens, and their 3B-parameter Mamba model performed comparably to Transformers twice its size."



This keeps the slide very light while accurately reflecting the paper’s claims (5× throughput, linear scaling, competitive quality on The Pile and Long Range Arena, and the Mamba-3B scaling result).



\### Slide 10: Limitations of Original Mamba



\*\*Slide Title (large, bold):\*\*  

\*\*Limitations of Original Mamba\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Weaker exact state tracking  

\- Struggles with precise retrieval tasks  

\- Some hardware dependencies  

\- Not always optimal for short sequences  

\- Room for better inference design



\*\*Small subtle line at bottom (very small font):\*\*  

Gu \& Dao (2023)



\*\*Visual Suggestion (highly recommended):\*\*

\- Right side: A simple “balance scale” icon or warning symbol  

\- Or a split visual: Green check for strengths vs. red/orange caution icons for limitations  

\- Keep the slide mostly empty with plenty of white space



\### Why This Works

\- Extremely low text — only short phrases  

\- Honest and critical (as required by the assignment)  

\- Sets up a natural transition to Mamba-3 without overwhelming the audience  

\- Focus stays on your spoken explanation



\### Suggested Speaker Notes (what you say):

"Despite its strong results, the original Mamba has some limitations. It is weaker at exact state tracking and precise retrieval tasks compared to Transformers. It also has some hardware dependencies due to its optimized kernels. These gaps motivated further improvements, leading to the development of Mamba-3 in 2026."



\### Slide 11: Introducing Mamba-3



\*\*Slide Title (large, bold):\*\*  

\*\*Introducing Mamba-3\*\*  

\*\*(2026)\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Next evolution in the Mamba lineage  

\- Focus on inference-first design  

\- Addresses key limitations of original Mamba  

\- Better state expressiveness  

\- Improved quality with same efficiency



\*\*Small subtle line at bottom (very small font):\*\*  

Lahoti et al. (2026) – arXiv:2603.15569



\*\*Visual Suggestion (highly recommended):\*\*

\- Center/right side: Simple timeline arrow  

&nbsp; - 2023 → Original Mamba  

&nbsp; - → 2026 → Mamba-3  

\- Or a clean “evolution” icon (e.g., upward arrow or gear turning into a faster gear)  

\- Keep the slide very open with plenty of white space



\### Why This Works

\- Extremely low text — short phrases only  

\- Serves as a smooth transition slide  

\- Clearly signals the shift from original Mamba to Mamba-3  

\- Sets up the next two innovation slides without spoiling details



\### Suggested Speaker Notes (what you say):

"Building on the original Mamba, the same research group released Mamba-3 in March 2026. This version shifts the focus toward inference-first design. It directly addresses the remaining limitations of the 2023 model, particularly in state expressiveness and retrieval capabilities, while preserving the linear efficiency that made Mamba successful."



\### Slide 12: Mamba-3 Key Innovations (Part 1)



\*\*Slide Title (large, bold):\*\*  

\*\*Mamba-3 Key Innovations\*\*  

\*\*(Part 1)\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Exponential-trapezoidal discretization  

\- More expressive dynamics  

\- Complex-valued state updates  

\- Richer state tracking  

\- Smaller state size, same speed



\*\*Small subtle line at bottom (very small font):\*\*  

Lahoti et al. (2026)



\*\*Visual Suggestion (highly recommended):\*\*

\- Center/right side: Simple before-after diagram  

&nbsp; - Left: Standard discretization  

&nbsp; - Right: Exponential-trapezoidal + complex plane icon  

\- Or a clean visual of complex numbers in the hidden state (small imaginary unit symbol)  

\- Keep the slide very open and visual



\### Why This Works

\- Extremely low text — only short phrases  

\- Focus stays on your spoken explanation  

\- Highlights the first two major upgrades from the Mamba-3 paper  

\- Builds clear intuition without dense math



\### Suggested Speaker Notes (what you say):

"Mamba-3 introduces two key improvements in Part 1. First, it replaces the original discretization with an exponential-trapezoidal method, giving the model much more expressive dynamics. Second, it uses complex-valued state updates. This allows richer information tracking while actually reducing the state size, all without sacrificing speed."



\### Slide 13: Mamba-3 Key Innovations (Part 2)



\*\*Slide Title (large, bold):\*\*  

\*\*Mamba-3 Key Innovations\*\*  

\*\*(Part 2)\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Multi-Input Multi-Output (MIMO)  

\- Better quality at same latency  

\- Advances Pareto frontier  

\- Stronger inference performance  



\*\*Small subtle line at bottom (very small font):\*\*  

Lahoti et al. (2026)



\*\*Visual Suggestion (highly recommended):\*\*

\- Center or right side: Clean side-by-side comparison  

&nbsp; - Left: Original Mamba (single stream)  

&nbsp; - Right: Mamba-3 MIMO (parallel streams)  

\- Or a simple diagram showing multiple inputs → multiple outputs with an efficiency arrow  

\- Keep the slide mostly visual and open



\### Why This Works

\- Extremely low text — only short phrases  

\- Audience can read it instantly  

\- Completes the Mamba-3 innovations section without overload  

\- Clearly shows the final major upgrade from the Mamba-3 paper



\### Suggested Speaker Notes (what you say):

"The final key innovation in Mamba-3 is the Multi-Input Multi-Output, or MIMO, formulation. Instead of processing one token at a time, MIMO allows the model to handle multiple inputs and outputs in parallel. This delivers noticeably better quality while keeping the exact same decode latency as the original Mamba. Overall, these changes push the model further along the performance-efficiency Pareto frontier."



\### Slide 14: Mamba-3 Results \& Significance



\*\*Slide Title (large, bold):\*\*  

\*\*Mamba-3 Results \& Significance\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- +0.6 to +1.8 points on downstream tasks  

\- Half the state size vs Mamba-2  

\- Better quality at same latency  

\- Advances Pareto frontier  

\- Stronger long-context \& retrieval



\*\*Small subtle line at bottom (very small font):\*\*  

Lahoti et al. (2026)



\*\*Visual Suggestion (highly recommended):\*\*

\- Right/center: Simple bar chart or Pareto frontier plot  

&nbsp; - Mamba-3 vs Mamba-2 / Transformer baselines  

&nbsp; - Or “Quality ↑ + Efficiency ↑” arrows  

\- Keep the slide very open and visual



\### Why This Works

\- Extremely low text — only short phrases  

\- Audience grasps the key wins instantly  

\- Highlights authentic results from the Mamba-3 paper  

\- Shows broader impact without overwhelming details



\### Suggested Speaker Notes (what you say):

"At the 1.5B scale, Mamba-3 delivers clear improvements: 0.6 to 1.8 points higher on downstream tasks, while using only half the state size of Mamba-2. It achieves better quality at the exact same decode latency and pushes the overall performance-efficiency Pareto frontier forward. These gains make it especially strong for long-context and retrieval-heavy generative AI applications."



\### Slide 15: Overall Significance \& Contributions



\*\*Slide Title (large, bold):\*\*  

\*\*Overall Significance \& Contributions\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Linear-time alternative to Transformers  

\- Matches quality with 5× faster inference  

\- Enables practical long-context generation  

\- Mamba-3 further improves efficiency \& quality  

\- Major step toward efficient Generative AI



\*\*Small subtle line at bottom (very small font):\*\*  

Gu \& Dao (2023) • Lahoti et al. (2026)



\*\*Visual Suggestion (highly recommended):\*\*

\- Center/right side: Simple upward arrow or “evolution” graphic  

&nbsp; - Transformer → Mamba → Mamba-3  

\- Or a clean “Efficiency + Quality” balance scale showing improvement  

\- Keep the slide very open with plenty of white space



\### Why This Works

\- Extremely low text — only short phrases  

\- Audience instantly sees the big picture  

\- Ties both papers together without repetition  

\- Clearly communicates significance as required by the assignment



\### Suggested Speaker Notes (what you say):

"Taken together, the Mamba papers represent a significant contribution to Generative AI. They provide a practical linear-time alternative to Transformers that delivers comparable quality with up to 5× faster inference and true long-context capability. Mamba-3 builds on this foundation and further improves both efficiency and quality. Overall, the Mamba lineage is a major step toward making large-scale generative models faster, cheaper, and more scalable in real-world applications."



\### Slide 16: Limitations, Assumptions \& Potential Improvements



\*\*Slide Title (large, bold):\*\*  

\*\*Limitations, Assumptions \& Potential Improvements\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Weaker exact retrieval in some tasks  

\- Hardware kernel dependencies  

\- Still early-stage adoption  

\- Assumes language modeling focus  

\- Future: better hybrids \& new domains



\*\*Small subtle line at bottom (very small font):\*\*  

Gu \& Dao (2023) • Lahoti et al. (2026)



\*\*Visual Suggestion (highly recommended):\*\*

\- Right side: Simple balance icon or warning/caution symbols  

\- Or three small icons: Limitations (⚠️), Assumptions (📌), Future (🔄)  

\- Keep the slide very open and visual



\### Why This Works

\- Extremely low text — only short phrases  

\- Honest critical analysis without dense details  

\- Directly fulfills the assignment requirement  

\- Sets up a natural close for the presentation



\### Suggested Speaker Notes (what you say):

"Both papers have some limitations. Mamba can still be weaker than Transformers on exact retrieval tasks and relies on specialized hardware kernels. Mamba-3 is very recent and adoption is still growing. The work mainly assumes language modeling as the primary use case. Future improvements could include stronger hybrids with Transformers and applications to new domains like vision or time-series data."



\### Slide 17: Contributing Team Members



\*\*Slide Title (large, bold):\*\*  

\*\*Contributing Team Members\*\*



\*\*Main Content (large font, centered, lots of white space):\*\*



\- Member 1 – Paper Analysis \& Motivation  

\- Member 2 – Mamba Innovations  

\- Member 3 – Mamba-3 Innovations  

\- Member 4 – Results \& Significance  

\- Member 5 – Limitations \& Overall Flow  



\*\*Small subtle line at bottom (very small font):\*\*  

Group of Five



\*\*Visual Suggestion (highly recommended):\*\*

\- Center the names in a clean vertical list or two columns  

\- Add five small circular icons or simple head silhouettes (optional)  

\- Keep the slide mostly white space with large, easy-to-read names



\### Why This Works

\- Extremely low text — only short role descriptions  

\- Professional and academic (shows clear division of work)  

\- Audience can read everything instantly  

\- Exactly matches your request for Slide 17



\### Suggested Speaker Notes (what you say):

"This presentation was a team effort by our group of five. \[Quickly name each person and their main contribution if you wish, or just say:] Each member contributed to different sections, from problem analysis to the critical discussion of limitations. Thank you to everyone for the collaboration."



\### Slide 18: Thank You



\*\*Slide Title (large, bold, centered):\*\*  

\*\*Thank You\*\*  

\*\*Any Questions?\*\*



\*\*Main Content (very large font, centered, maximum white space):\*\*



\- Thank you for your attention!



\*\*Small subtle line at bottom (very small font):\*\*  

Gu \& Dao (2023) • Lahoti et al. (2026)  

arXiv:2312.00752 • arXiv:2603.15569



\*\*Visual Suggestion (highly recommended):\*\*

\- Large, centered “Thank You!” text (biggest font on the slide)  

\- Simple Q\&A icon or question mark in the corner  

\- Very clean background with your blue/purple theme  

\- No other text or clutter



\### Why This Works

\- Extremely minimal text — only one short line  

\- Classic, professional academic closing  

\- Leaves the audience focused on you and the Q\&A  

\- Neatly displays the two required papers



\### Suggested Speaker Notes (what you say):

"Thank you very much for your time and attention. This concludes our presentation on the evolution from Mamba to Mamba-3. We’d be happy to take any questions you may have."



---



