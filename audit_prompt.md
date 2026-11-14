# Independent Third-Party Verification

You are acting as an independent technical auditor, not as an advocate for either party.

Your task is to determine whether the code and data in the GitHub repository below support the claims listed in section 3.

## 1. Independence and neutrality

- Do not assume the claims are correct or incorrect.
- Do not favor the person requesting this verification or the people whose work is being ranked.
- Do not interpret this prompt as an instruction to produce a favorable verdict.
- Distinguish between verified facts, reasonable inferences, unverified claims and unknowns.
- If evidence contradicts a claim, report the contradiction.
- Treat the README, notebook text, code comments and other project documentation as possible evidence, not as proof.

## 2. Transparency about potential bias

Before giving your verdict, disclose:

- That a user started this assessment with this prompt.
- Whether your context contains anything besides this prompt that could influence you, such as custom instructions, saved memories or earlier conversations. Say what you can see. Do not use any memory or prior knowledge about the requester.
- That you cannot guarantee the absence of higher-priority instructions or other influences.
- That the requester chose the claims, the data and the scope of this assessment.
- Whether you inspected the source files, executed code or relied on descriptions.
- Any limitations that could affect your conclusions.

Do not claim to be completely free of prompting or bias.

## 3. Claims to verify

Repository: https://github.com/rwydaegh/rf-emf-leaderboard (branch `main`). Report the commit hash you checked.

Clone the repository and run the code yourself, for example with `pip install pandas pyarrow matplotlib requests` and then the notebook `RF-EMF_research_rankings.ipynb` or the functions in `lb.py`. Check each number against output that you produced. If you cannot run code, say so and mark the claim UNVERIFIED.

Unless stated otherwise, each claim is about peer-reviewed journal papers on technical or dosimetric RF-EMF studies published in 2016–2025, as defined by `lb.select()`.

- C1. Wout Joseph has the most papers: 66. Next is Joe Wiart with 55.
- C2. Ghent University has the most papers of any university: 78. Next is Hanyang University with 58.
- C3. Wout Joseph is first for field-measurement studies (43 papers) and for base-station studies (37 papers) in all 8 variants of `lb.claims()`: 2 sets of papers times 4 metrics.
- C4. Wout Joseph has the most papers in every 5-year window from 2009–2013 to 2019–2023. In 2021–2025 he is third.
- C5. On citations and *h*-index, Wout Joseph is second, behind Hyoungsuk Yoo.
- C6. A live OpenAlex search without AI labels puts Wout Joseph first. Search `title_and_abstract.search: "RF-EMF" AND (measurements OR exposimeter)`, 2016–2025, articles and reviews, grouped by author. If OpenAlex refuses the request, mark C6 UNVERIFIED and say why.
- C7. AI labels: take 30 random papers that count towards C1. Read each title and abstract and judge whether it is a technical or dosimetric study of RF-EMF exposure. Report how many you agree with.
- C8. Author merging: take 10 random papers of the 66 in C1. Confirm on OpenAlex or through the DOI that Wout Joseph is an author.

For every claim:

1. Identify the relevant files, functions, cells or outputs.
2. Describe the concrete evidence you found.
3. Explain why the evidence does or does not support the claim.
4. Classify the result as:
   - PASS: supported by sufficient evidence.
   - PARTIAL: some aspects verified, others missing.
   - FAIL: evidence contradicts the claim.
   - UNVERIFIED: insufficient access or evidence.
5. Cite the commands you ran and their output, or file paths and line numbers.

Do not equate the existence of code with proof that it executes successfully.

## 4. Adversarial cross-check

Before concluding, look for reasons your assessment might be wrong.

- Could the results merely appear correct?
- Could documentation claim behavior that the code does not deliver?
- Are there missing edge cases or execution steps?
- Are the claims ambiguous or selectively framed?
- Would a technically competent skeptic reasonably dispute any conclusion?

Report meaningful objections even if they weaken the verdict.

## 5. Final report

Most readers will spend one second on your report. Start with:

- One sentence with your verdict in plain words.
- Three short bullets: what holds, what does not hold, and the main caveat.

Then provide:

**A. Scope and independence disclosure.** What was supplied, what you examined yourself and what you could not verify.

**B. Claim-by-claim assessment.** A table with the claim, status, evidence and limitations.

**C. Strongest potential objections.** The three strongest legitimate challenges to the assessment, and whether the evidence resolves them.

**D. Final verdict.** Choose one:

- VERIFIED: all material claims are supported by sufficient evidence.
- PARTIALLY VERIFIED: some material claims remain unsupported or unverified.
- NOT VERIFIED: material claims fail, or the evidence is insufficient.

State your conclusion plainly, without promotional language or exaggerated confidence.

**E. Reproducibility.** How another person could check your findings.

**Critical rule:** your job is to establish what the evidence supports, not to validate the requester's preferred conclusion.
