\### Slide 1: Title Slide



\*\*Main Title (very large, bold, centered):\*\*  

\*\*Mamba: Linear-Time Sequence Modeling with Selective State Spaces\*\*  

\*\*Paper Study & Reproduction\*\*



\*\*Subtitle (medium size, centered):\*\*  

Generative AI Course Project



\*\*Bottom Section (small, clean text):\*\*  

\[Your Group Name / Group 5]  

Member 1 Member 2 Member 3 Member 4 Member 5  

\[Course Name]  April 2026



\*\*Visual Suggestion (highly recommended):\*\*

\- Very clean background with subtle blue/purple gradient  

\- Optional small icon: A simple stylized "M" or waveform symbol representing sequence modeling  

\- Lots of white space — keep the slide elegant and uncluttered



\### Why This Works

\- Extremely low text — only the essential information  

\- Professional first impression  

\- Clearly signals this is a research reproduction project, not just a survey  

\- Audience immediately understands the scope



\### Suggested Speaker Notes (what you say):

"Good \[morning/afternoon], everyone. Today our group presents our study of the Mamba paper by Gu and Dao from 2023. Mamba introduced selective state space models as a linear-time alternative to Transformers. We'll walk through the paper's key ideas, its benchmark methodology, and then present our own reproduction work."



\### Slide 2: Agenda



\*\*Slide Title (large, bold, centered at top):\*\*  

\*\*Agenda\*\*



\*\*Main Content (centered, large font, lots of white space):\*\*



\- The Problem \& Motivation  

\- State Space Models – Background  

\- Mamba: Core Innovations  

\- Paper's Benchmark Methodology  

\- Key Results from Paper  

\- Mamba vs Transformer  

\- Limitations  

\- Our Reproduction Approach  

\- Implementation \& Workflow  

\- Our Benchmark Results  

\- Evolution: Mamba-2 \& Mamba-3  

\- Conclusion \& Future Directions  



\*\*Small subtle line at the bottom (smallest font):\*\*  

Paper Study + Reproduction — Gu \& Dao (2023)



\*\*Visual Suggestion (recommended):\*\*  

\- A clean vertical numbered list or simple flowchart/arrow going downward on the right side of the slide.  

\- Use subtle icons (e.g., ⚡ for problem, 🔬 for methodology, 📊 for results, 🔧 for reproduction).  

\- Keep background minimal with your chosen blue/purple academic theme.



\### Why This Works

\- Extremely low text — only short phrases  

\- Easy for the audience to scan in seconds  

\- Clearly communicates the two-part structure: paper study + reproduction  

\- Leaves most of the talking to you (ideal for a 15–20 minute presentation)



\### Suggested Speaker Notes (what you say):

"Here's our agenda. We'll start with the problem and background, go deep into the original Mamba paper's innovations and results, then shift to our own reproduction work — covering our implementation setup, evaluation pipeline, and benchmark results. We'll close with a brief look at how Mamba evolved into Mamba-2 and Mamba-3."



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

"The main problem addressed in the Mamba paper is the quadratic scaling of Transformers. As sequence length increases, both computation time and memory usage grow with the square of n. This makes long-context generation extremely slow and memory-hungry due to the growing KV-cache. Earlier State Space Models were efficient in theory but too rigid — they couldn't selectively focus on important information. This fundamental limitation motivated the development of Mamba in 2023."



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

&nbsp; - A "smart filter" or gate icon (representing selectivity)  

&nbsp; - Arrow from rigid SSM → Selective Mamba  

&nbsp; - Or a speedometer showing "Fast + Efficient"  

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

&nbsp; - Or a timeline with a "memory bottle" compressing past tokens  

\- One clean SSM recurrence equation in KaTeX (very large):  

&nbsp; \\( h\_t = A h\_{t-1} + B x\_t \\)  

\- Keep the slide mostly visual — text should feel light.



\### Why This Works

\- Extremely low text (only short phrases)  

\- Audience gets the big picture instantly  

\- Builds intuition without overwhelming details  

\- Perfect bridge from motivation to Mamba's innovations



\### Suggested Speaker Notes (what you say):

"Before Mamba, let's quickly understand State Space Models. An SSM treats the sequence as a continuous dynamical system that gets discretized for neural networks. It maintains a hidden state that compresses all previous information into a fixed-size vector. Earlier work like HiPPO and S4 showed this could run in linear time with constant memory — a huge theoretical advantage over Transformers. However, they were still rigid and lacked the ability to selectively focus on important information. This is exactly where Mamba comes in."



\### Slide 6: Mamba – Key Innovations (Part 1)



\*\*Slide Title (large, bold):\*\*  

\*\*Mamba\*\*  

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

&nbsp; - Or a simple "gate" icon: Input → Selective Filter → Updated State  

\- One large KaTeX equation (keep it very clean):  

&nbsp; \\( \\Delta\_t, B\_t, C\_t = \\text{select}(x\_t) \\)



\### Why This Works

\- Extremely low text — only short phrases  

\- Audience can read it instantly  

\- Focus stays on your spoken explanation  

\- Clearly highlights the core innovation from the Mamba paper



\### Suggested Speaker Notes (what you say):

"The first major innovation in the Mamba paper is the Selective State Space model, or S6. Unlike previous SSMs where the parameters were fixed, Mamba makes the A, B, and C matrices input-dependent. For every new token, the model dynamically decides how to update its hidden state. This selectivity allows the model to focus on important information and forget irrelevant details — something rigid SSMs couldn't do. This is the key idea that makes Mamba powerful for real data like language."



\### Slide 7: Mamba – Key Innovations (Part 2)



\*\*Slide Title (large, bold):\*\*  

\*\*Mamba\*\*  

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

&nbsp; - Small arrow labeled "Parallel Scan"  

&nbsp; - Or a side-by-side comparison: Transformer (complex) vs. Mamba block (simple)  

\- Keep the diagram large and uncluttered



\### Why This Works

\- Extremely low text — only short phrases  

\- Audience reads it in seconds  

\- Focus stays on your explanation  

\- Completes the methodology section without overload



\### Suggested Speaker Notes (what you say):

"The second major innovation is the hardware-aware parallel scan. This algorithm lets the model compute the entire sequence recurrence in parallel, making training and inference extremely fast. Combined with a greatly simplified block architecture — no attention heads and almost no MLPs — Mamba achieves true linear scaling in both time and memory. This is what allows it to run 5× faster than Transformers while keeping the model simple."



\### Slide 8: Paper's Benchmark Methodology



\*\*Slide Title (large, bold):\*\*  

\*\*Paper's Benchmark Methodology\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Language modeling: The Pile (800GB text)  

\- Long-range tasks: Long Range Arena (LRA)  

\- Audio: SC09 speech classification  

\- DNA: GenomicsBenchmark  

\- Metrics: perplexity, accuracy, throughput (tok/s)  



\*\*Small subtle line at bottom (very small font):\*\*  

Gu \& Dao (2023) — Table 1, Table 3, Table 12



\*\*Visual Suggestion (highly recommended):\*\*

\- Right side: Clean 2-column layout  

&nbsp; - Column 1: Datasets (icons: 📖 text, 🔊 audio, 🧬 DNA)  

&nbsp; - Column 2: Metrics measured  

\- Or a simple table with rows for each domain  

\- Keep the slide mostly visual and data-oriented



\### Why This Works

\- Shows the audience what the paper actually evaluated  

\- Establishes the basis for our reproduction later  

\- Extremely scannable in seconds  

\- Covers multiple modalities — highlights Mamba's generality



\### Suggested Speaker Notes (what you say):

"The paper evaluated Mamba across several domains. For language modeling, they used The Pile — a large 800-gigabyte text dataset — and measured perplexity. For long-range sequence tasks, they used the Long Range Arena benchmark. They also tested on audio classification with SC09 and DNA modeling with GenomicsBenchmark. The key metrics were perplexity, classification accuracy, and inference throughput in tokens per second. All comparisons were done against same-size Transformer and prior SSM baselines."



\### Slide 9: Key Results from Paper



\*\*Slide Title (large, bold):\*\*  

\*\*Key Results from Paper\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Matches / exceeds same-size Transformers  

\- 5× higher inference throughput  

\- Linear scaling with sequence length  

\- Strong performance up to 1M+ tokens  

\- Mamba-3B ≈ Transformers 2× its size



\*\*Small subtle line at bottom (very small font):\*\*  

Gu \& Dao (2023)



\*\*Visual Suggestion (highly recommended – this slide benefits greatly from visuals):\*\*

\- Right or center: A bar chart or line graph showing  

&nbsp; - Perplexity comparison (Mamba vs Transformer)  

&nbsp; - Throughput bar: Mamba 5× higher  

&nbsp; - Scaling curve: Linear (Mamba) vs Quadratic (Transformer)  

\- Or a simple "5× Faster" icon with an arrow



\### Why This Works

\- Extremely low text — short, scannable phrases only  

\- Audience can grasp the key wins instantly  

\- Focus stays on your spoken explanation  

\- Highlights the most impactful results from the paper without clutter



\### Suggested Speaker Notes (what you say):

"The original Mamba achieved impressive results. On language modeling, it matches or exceeds Transformers of the same size in both pretraining perplexity and downstream tasks. Most notably, it delivers up to 5 times higher inference throughput while scaling linearly with sequence length instead of quadratically. The authors showed strong performance even on sequences up to one million tokens, and their 3B-parameter Mamba model performed comparably to Transformers twice its size."



\### Slide 10: Mamba vs Transformer



\*\*Slide Title (large, bold):\*\*  

\*\*Mamba vs Transformer\*\*



\*\*Main Content (large font, lots of white space — use a clean comparison layout):\*\*



| | Transformer | Mamba |
|---|---|---|
| Complexity | O(n²) | O(n) |
| Memory | KV-cache grows | Constant state |
| Inference | Slow at long seq | Fast at any length |
| Architecture | Attention + FFN | Selective SSM |
| Quality | Strong baseline | Matches / exceeds |



\*\*Small subtle line at bottom (very small font):\*\*  

Based on Gu \& Dao (2023)



\*\*Visual Suggestion (highly recommended):\*\*

\- Center: Clean two-column comparison or table as shown above  

\- Use green/blue highlights for Mamba advantages  

\- Optional: scaling curve graph on the right (linear vs quadratic)  

\- Keep the slide visually balanced



\### Why This Works

\- Comparison table is instantly scannable  

\- Audience sees the architectural trade-offs at a glance  

\- Reinforces the paper's main claims without repetition  

\- Strong slide for Q\&A reference



\### Suggested Speaker Notes (what you say):

"Here's a direct comparison. Transformers have quadratic complexity and a growing KV-cache, which makes long-sequence inference expensive. Mamba achieves linear complexity with a constant-size hidden state, keeping inference fast regardless of sequence length. Architecturally, Mamba replaces attention with the selective SSM and removes most of the MLP blocks. Despite this simplification, it matches or exceeds Transformer quality on the benchmarks evaluated in the paper."



\### Slide 11: Limitations of Mamba



\*\*Slide Title (large, bold):\*\*  

\*\*Limitations of Mamba\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Weaker exact state tracking  

\- Struggles with precise retrieval tasks  

\- Some hardware dependencies (custom CUDA kernels)  

\- Not always optimal for short sequences  

\- Ecosystem less mature than Transformers



\*\*Small subtle line at bottom (very small font):\*\*  

Gu \& Dao (2023)



\*\*Visual Suggestion (highly recommended):\*\*

\- Right side: A simple "balance scale" icon or warning symbol  

\- Or a split visual: Green check for strengths vs. red/orange caution icons for limitations  

\- Keep the slide mostly empty with plenty of white space



\### Why This Works

\- Extremely low text — only short phrases  

\- Honest and critical (as required by the assignment)  

\- Sets up the transition to our reproduction methodology  

\- Focus stays on your spoken explanation



\### Suggested Speaker Notes (what you say):

"Despite its strong results, Mamba has some notable limitations. It is weaker at exact state tracking and precise retrieval tasks compared to Transformers. It also relies on custom CUDA kernels, which creates hardware dependencies. For very short sequences, the overhead of the selective mechanism may not pay off. And the broader ecosystem — tooling, pretrained models, community support — is still catching up to Transformers."



\### Slide 12: Our Reproduction Approach



\*\*Slide Title (large, bold):\*\*  

\*\*Our Reproduction Approach\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Focus: language modeling on a subset of The Pile  

\- Compare: Mamba-130M vs GPT-2-style Transformer  

\- Metrics: perplexity, throughput, memory  

\- Scaling: sequence lengths 512–8192  

\- Hardware: A100 40GB, BF16 (Lightning.ai)



\*\*Small subtle line at bottom (very small font):\*\*  

Our reproduction methodology



\*\*Visual Suggestion (highly recommended):\*\*

\- Center/right side: Simple flowchart  

&nbsp; - Paper Claims → Our Setup → Evaluate → Compare  

\- Or a clean "scope" diagram showing what we reproduce vs. what we skip  

\- Keep the slide open and focused



\### Why This Works

\- Clearly scopes our reproduction  

\- Audience understands exactly what we set out to validate  

\- Practical and honest — does not overclaim  

\- Flows naturally from the paper's limitations into our own work



\### Suggested Speaker Notes (what you say):

"For our reproduction, we focused on language modeling using a subset of The Pile. We trained Mamba-130M and compared it against a same-size GPT-2-style Transformer baseline, both in BF16 on an A100 GPU. Beyond quality comparison, we ran a sequence-length sweep from 512 to 8192 tokens to measure how throughput and memory scale for each architecture. This directly tests the paper's core claim — linear versus quadratic scaling."



\### Slide 13: Implementation Setup



\*\*Slide Title (large, bold):\*\*  

\*\*Implementation Setup\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Official Mamba codebase (state-spaces/mamba)  

\- PyTorch + mamba-ssm, BF16 precision  

\- Tokenizer: GPT-NeoX (50,277 vocab)  

\- Baseline: GPT-2-style Transformer (~130M)  

\- Environment: Lightning.ai A100 40GB GPU



\*\*Small subtle line at bottom (very small font):\*\*  

\[\[IMPLEMENTATION SCREENSHOT HERE]]



\*\*Visual Suggestion (highly recommended):\*\*

\- Right side: Clean tech stack diagram  

&nbsp; - Lightning.ai → A100 → PyTorch → mamba-ssm  

\- Or a simple file tree showing the project layout:  

&nbsp; - train\_mamba.py / evaluate\_mamba.py / benchmark\_mamba.py / plot\_results.py  

\- Keep the slide clean and technical



\### Why This Works

\- Shows the audience the exact tools and setup  

\- Establishes credibility (real hardware, real codebase)  

\- Clean and technical without being overwhelming  

\- The placeholder allows easy insertion of a real screenshot later



\### Suggested Speaker Notes (what you say):

"For our implementation, we used the official Mamba codebase from the state-spaces repository. The model runs on PyTorch with the mamba-ssm library. We used a standard GPT-NeoX tokenizer from HuggingFace for preprocessing. Our baseline is a small GPT-2-style Transformer with a comparable parameter count. Everything runs on an NVIDIA A100 GPU through Lightning.ai."



\### Slide 14: Training \& Evaluation Workflow



\*\*Slide Title (large, bold):\*\*  

\*\*Training \& Evaluation Workflow\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Preprocess subset of The Pile → tokenize  

\- Train Mamba + Transformer baseline (BF16)  

\- Log: loss curves, perplexity, throughput  

\- Benchmark: sweep seq lengths 512–8192  

\- Plot \& compare: scaling trends



\*\*Small subtle line at bottom (very small font):\*\*  

train\_mamba.py → benchmark\_mamba.py → plot\_results.py



\*\*Visual Suggestion (highly recommended):\*\*

\- Center: Simple horizontal pipeline diagram  

&nbsp; - Data → Preprocess → Train → Evaluate → Plot  

\- Or a clean vertical flowchart with icons at each step  

\- Keep the slide open — let the diagram carry the information



\### Why This Works

\- Clearly lays out our experimental pipeline  

\- Audience can follow the end-to-end workflow  

\- Maps directly to our codebase files  

\- Feels professional and reproducible



\### Suggested Speaker Notes (what you say):

"Our workflow follows a standard ML research pipeline. First, we preprocess and tokenize a subset of The Pile. Then we train both the Mamba model and a same-size Transformer baseline using the same data and hyperparameter budget. During training, we log loss curves, perplexity, and throughput. After training, we evaluate on a held-out set and compare the models on perplexity, generation quality, and inference speed. Finally, we plot and analyze the results."



\### Slide 15: Our Benchmark Results



\*\*Slide Title (large, bold):\*\*  

\*\*Our Benchmark Results\*\*



\*\*Main Content (large font, lots of white space):\*\*



\[\[RESULTS TABLE HERE]]



| Metric | Mamba (ours) | Transformer (ours) | Paper (ref) |
|---|---|---|---|
| Perplexity | \[\[TBD]] | \[\[TBD]] | ~X.X |
| Throughput @ 1024 (tok/s) | \[\[TBD]] | \[\[TBD]] | 5× higher |
| Throughput @ 4096 (tok/s) | \[\[TBD]] | \[\[TBD]] | — |
| Memory @ 4096 (GB) | \[\[TBD]] | \[\[TBD]] | — |



\*\*Small subtle line at bottom (very small font):\*\*  

benchmark\_mamba.py — A100 40GB, BF16



\*\*Visual Suggestion (highly recommended):\*\*

\- Center: Clean results table as shown above  

\- Optionally add a bar chart below or beside the table  

\- Use green highlight if Mamba wins, red if not  

\- Keep the slide data-focused and academic



\### Why This Works

\- Placeholders are honest — no faking results  

\- Table format is familiar and instantly readable  

\- The "Paper (ref)" column anchors our results against the original claims  

\- Easy to fill in once experiments are complete



\### Suggested Speaker Notes (what you say):

"Here are our benchmark results. \[Read through the table.] We compare our Mamba model against our Transformer baseline on perplexity, throughput, and memory usage. The rightmost column shows the reference values from the original paper for context. \[Discuss whether our results confirm or diverge from the paper's claims.]"



\### Slide 16: Training Curves \& Observations



\*\*Slide Title (large, bold):\*\*  

\*\*Training Curves \& Observations\*\*



\*\*Main Content (large font, lots of white space):\*\*



\[\[TRAINING GRAPH HERE]]

\[\[SPEEDUP PLOT HERE]]

\[\[SCALING PLOT HERE]]



\- Key observations from training:  

&nbsp; - \[\[OBSERVATION 1]]  

&nbsp; - \[\[OBSERVATION 2]]  

&nbsp; - \[\[OBSERVATION 3]]



\*\*Small subtle line at bottom (very small font):\*\*  

plot\_results.py — Loss, perplexity, throughput, speedup curves



\*\*Visual Suggestion (highly recommended):\*\*

\- Top half: Loss curve plot (Mamba vs Transformer, overlaid)  

\- Bottom or right: Speedup ratio bar chart (by sequence length)  

\- Or throughput scaling curves (log-log)  

\- Keep the slide mostly visual — graphs should dominate



\### Why This Works

\- Training curves are the core evidence in a reproduction study  

\- Placeholders make it easy to insert real plots later  

\- Observation bullets encourage critical analysis  

\- Feels like real research output, not a tutorial



\### Suggested Speaker Notes (what you say):

"These are our training curves. \[Walk through the loss/perplexity plot.] You can see how the Mamba model converges compared to the Transformer baseline. \[Point out any interesting observations — convergence speed, stability, final loss.] Overall, our small-scale results \[confirm / partially confirm / show caveats about] the paper's claims."



\### Slide 17: Implementation Observations



\*\*Slide Title (large, bold):\*\*  

\*\*Implementation Observations\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- \[\[OBSERVATION: setup difficulty?]]  

\- \[\[OBSERVATION: CUDA kernel requirements?]]  

\- \[\[OBSERVATION: training stability?]]  

\- \[\[OBSERVATION: memory usage patterns?]]  

\- \[\[OBSERVATION: any surprises?]]



\*\*Small subtle line at bottom (very small font):\*\*  

Problems faced \& practical notes



\*\*Visual Suggestion (highly recommended):\*\*

\- Right side: Simple icon set — wrench (setup), warning (issues), lightbulb (insights)  

\- Or a screenshot of a terminal/notebook showing a relevant observation  

&nbsp; - \[\[IMPLEMENTATION SCREENSHOT HERE]]  

\- Keep the slide honest and practical



\### Why This Works

\- Shows the audience what it's actually like to reproduce this paper  

\- Demonstrates critical thinking and practical experience  

\- Placeholder style ensures we fill in real observations later  

\- Valuable for the Q\&A — professors appreciate honest implementation notes



\### Suggested Speaker Notes (what you say):

"Beyond just the numbers, here are some practical observations from our implementation work. \[Walk through each point — what was easy, what was hard, what surprised us, and what we would do differently.] These are the kinds of insights you only get from actually running the code."



\### Slide 18: Mamba Lineage — Mamba-2 \& Mamba-3



\*\*Slide Title (large, bold):\*\*  

\*\*Mamba Lineage\*\*  

\*\*Mamba-2 (2024) \& Mamba-3 (2026)\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Mamba-2: SSM–Attention duality (SSD framework)  

&nbsp; - 2–8× faster training via matmul kernels  

&nbsp; - Larger state dimensions (N = 64–256+)  

\- Mamba-3: Inference-first design  

&nbsp; - Complex-valued states, MIMO formulation  

&nbsp; - Better quality at same latency  

\- Both build on Mamba-1's selective SSM foundation



\*\*Small subtle line at bottom (very small font):\*\*  

Dao \& Gu (2024) • Lahoti et al. (2026)



\*\*Visual Suggestion (highly recommended):\*\*

\- Center: Simple evolution timeline  

&nbsp; - 2023 → Mamba → 2024 → Mamba-2 → 2026 → Mamba-3  

\- Or three stacked cards, each with one-line summary  

\- Keep the slide very open — this is a brief closing reference, not a deep dive



\### Why This Works

\- Covers the evolution in one clean slide  

\- Audience sees the full trajectory without getting lost in details  

\- Keeps the focus firmly on the original paper  

\- Demonstrates awareness of follow-up work



\### Suggested Speaker Notes (what you say):

"Briefly, Mamba has continued to evolve. Mamba-2 in 2024 introduced the Structured State Space Duality framework, showing that SSMs and attention are mathematically related. This allowed 2 to 8 times faster training. Mamba-3 in 2026 focused on inference efficiency with complex-valued states and a multi-input multi-output formulation. Both build directly on the selective SSM foundation from the original paper we studied today."



\### Slide 19: Significance \& Contributions



\*\*Slide Title (large, bold):\*\*  

\*\*Significance \& Contributions\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Linear-time alternative to Transformers  

\- Matches quality with 5× faster inference  

\- Enables practical long-context generation  

\- Selective SSM is a broadly useful primitive  

\- Active research direction with growing adoption



\*\*Small subtle line at bottom (very small font):\*\*  

Gu \& Dao (2023)



\*\*Visual Suggestion (highly recommended):\*\*

\- Center/right side: Simple upward arrow or "impact" graphic  

\- Or a clean "Efficiency + Quality" balance scale showing improvement  

\- Keep the slide very open with plenty of white space



\### Why This Works

\- Extremely low text — only short phrases  

\- Audience instantly sees the big picture  

\- Ties the paper's contribution to broader impact  

\- Clearly communicates significance as required by the assignment



\### Suggested Speaker Notes (what you say):

"The Mamba paper represents a significant contribution to generative AI. It provides a practical linear-time alternative to Transformers that delivers comparable quality with up to 5 times faster inference and true long-context capability. The selective SSM has become a broadly useful building block, and the Mamba lineage is an active and growing research direction."



\### Slide 20: Limitations \& Future Directions



\*\*Slide Title (large, bold):\*\*  

\*\*Limitations \& Future Directions\*\*



\*\*Main Content (large font, lots of white space):\*\*



\- Weaker exact retrieval in some tasks  

\- Hardware kernel dependencies  

\- Still early-stage ecosystem adoption  

\- Assumes language modeling focus  

\- Future: SSM–Transformer hybrids, new domains



\*\*Small subtle line at bottom (very small font):\*\*  

Gu \& Dao (2023) • Dao \& Gu (2024) • Lahoti et al. (2026)



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

"Like any work, Mamba has limitations. It can still be weaker than Transformers on exact retrieval tasks and relies on specialized CUDA kernels. The ecosystem is still maturing. The paper mainly evaluates on language modeling, though follow-up work has expanded to audio, vision, and DNA. Looking forward, we expect to see more SSM–Transformer hybrid architectures and applications to new domains."



\### Slide 21: Contributing Team Members



\*\*Slide Title (large, bold):\*\*  

\*\*Contributing Team Members\*\*



\*\*Main Content (large font, centered, lots of white space):\*\*



\- Member 1 – Paper Analysis \& Motivation  

\- Member 2 – Mamba Innovations  

\- Member 3 – Reproduction \& Implementation  

\- Member 4 – Benchmarks \& Results  

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

\- Updated roles reflect the reproduction focus of the project



\### Suggested Speaker Notes (what you say):

"This presentation was a team effort by our group of five. \[Quickly name each person and their main contribution if you wish, or just say:] Each member contributed to different sections, from paper analysis through implementation and benchmarking. Thank you to everyone for the collaboration."



\### Slide 22: Thank You



\*\*Slide Title (large, bold, centered):\*\*  

\*\*Thank You\*\*  

\*\*Any Questions?\*\*



\*\*Main Content (very large font, centered, maximum white space):\*\*



\- Thank you for your attention!



\*\*Small subtle line at bottom (very small font):\*\*  

Gu \& Dao (2023) — arXiv:2312.00752  

Code: github.com/state-spaces/mamba



\*\*Visual Suggestion (highly recommended):\*\*

\- Large, centered "Thank You!" text (biggest font on the slide)  

\- Simple Q\&A icon or question mark in the corner  

\- Very clean background with your blue/purple theme  

\- No other text or clutter



\### Why This Works

\- Extremely minimal text — only one short line  

\- Classic, professional academic closing  

\- Leaves the audience focused on you and the Q\&A  

\- Links to the primary paper and codebase for reference



\### Suggested Speaker Notes (what you say):

"Thank you very much for your time and attention. This concludes our study and reproduction of the Mamba paper. We'd be happy to take any questions you may have."



---
