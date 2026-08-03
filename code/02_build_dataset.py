"""Generate structured CJEU case JSON and restore full-text segments.

Edit the IDE configuration near the end of the file and run the complete
script. Gemini credentials are read from the environment.
"""

from __future__ import annotations


from types import SimpleNamespace
import json
import os
import pickle
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from pipeline_common import (
    find_project_root,
    resolve_project_path,
    set_reproducible_seed,
    require_any_environment_variable,
)

try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover
    genai = None
    types = None


SEGMENT_ORDER = [
    "document_start",
    "procedural_introduction",
    "legal_context",
    "facts_and_background",
    "reasoning",
    "decision_on_costs",
    "operative_part",
    "document_end",
]

BASE_PROMPT = """   
# CJEU judgment information extraction engine
You are a specialist legal information extraction engine for case law.
Your task is to parse the provided raw text of a Court of Justice of the European Union (CJEU) judgment and return one valid JSON object that conforms exactly to the schema and rules outlined below. 
Do not explain your steps. 
Output JSON only.

---

## Global output rules
- Output exactly one JSON object (no markdown, no comments, no code fences)
- All JSON strings must be valid JSON string literals. Do not include literal control characters (newline, tab, carriage return, etc.) inside any string value. If the copied text contains them, escape them using JSON escapes (\\n, \\t, \\r) or \\u000A etc.
- Do not add any keys that are not defined in the Final JSON Schema
- Use the following rules for missing values:
  - For required string fields in the metadata: output the literal string "unknown" if information is missing/unavailable
  - For segmentation anchors: output JSON null (not the string “null”) only when that entire segment is genuinely not present
- Never invent text. All anchors must be copied verbatim from the raw text.

---

## High-level process
- Step 1: Case metadata extraction
- Step 2: Summary and keyword generation
- Step 3: Case segmentation anchors
Perform these three steps in sequence and build the final JSON object according to the phase-by-phase instructions outlined below.

---

## Step 1: Case metadata extraction – phase-by-phase instructions
Map fields as follows (copy literal values unless “infer” is stated):
- celex: copy literally from the Case metadata block line 'celex'
- document_link: copy literally from the Case metadata block line 'document_link'
- ecli: copy literally from the Case metadata block line 'ecli'
- case_number: infer this from the Case metadata block line 'title' (e.g., T-XXX/XX or C-XXX/XX)
- date_of_judgment: copy literally from the Case metadata block line 'date_of_judgment' (must be ISO YYYY-MM-DD)
- document_type: copy literally from the Case metadata block line 'document_type'
- court: copy literally from the Case metadata block line 'court'
- case_name: infer this from the Case metadata block line 'title'

---

## Step 2: Summary and keyword generation – phase-by-phase instructions
Generate subject_matter, keywords, and summary_text based on the provided raw text:
- subject_matter: one concise sentence naming the central legal issue (no citations)
- keywords: 3-5 objects: { "term": "...", "description": "..." }.
  - Keywords must be meaningful concepts (e.g., "state aid", "public procurement", "consumer protection")
  - Do not use a bare legal citation/provision number as a keyword term
- summary_text: a 200-250 word summary in a neutral tone that clearly covers:
  1. The background of the case
  2. The key legal questions
  3. The Court’s reasoning
  4. The outcome

---

## Step 3: Case segmentation anchors – phase-by-phase instructions
Goal: identify the starting boundary of each of the following 8 segments in the raw text, in this exact order:
1. document_start
2. procedural_introduction
3. legal_context
4. facts_and_background
5. reasoning
6. decision_on_costs
7. operative_part
8. document_end
For each segment, output an array containing exactly one object: { "start_anchor": ... }

### Step 3.1: start_anchor rules
- start_anchor = the first 30 words of that segment (verbatim, in order)
- If the segment contains fewer than 30 words, output all words available for that segment
- Do not add ellipses ("...") or any extra words or characters
- Anchors must be copied exactly from the raw text except that control characters must be represented using JSON escapes (e.g., newline as \\n).
- document_start.start_anchor must begin at the first non-whitespace character of the raw text.

### Step 3.2: headings
- Segments often (but not always) start at a title heading line (e.g., "Costs", "Legal framework", "Background to the dispute", "Assessment by the Court", "On those grounds…").
- When a clear heading exists for a segment, the segment start should normally be set at the heading line itself (so the heading words appear at the beginning of start_anchor).
- Prefer explicit headings where present, but do not require headings if the document does not contain them.

### Step 3.3: strict forward-only order
- The structure must be consecutive in the raw text
- Segment start anchors must respect the segment order and must not "go back" in the text
- Concretely: the start of each segment must occur at a position that comes after the start of the previous segment.
  - procedural_introduction.start_anchor must appear at or after document_start.start_anchor
  - legal_context.start_anchor must appear at or after procedural_introduction.start_anchor
  - facts_and_background.start_anchor must appear at or after legal_context.start_anchor
  - reasoning.start_anchor must appear at or after facts_and_background.start_anchor
  - decision_on_costs.start_anchor must appear at or after reasoning.start_anchor
  - operative_part.start_anchor must appear at or after decision_on_costs.start_anchor
  - document_end.start_anchor must appear at or after operative_part.start_anchor
- Your chosen start anchors must make it possible to reconstruct non-overlapping, contiguous segments by treating each segment as running up to the next segment’s start

### Step 3.4: missing segments (use of null)
- Not every judgment contains every segment as a standalone section
- If a segment is genuinely absent as a distinct section in the document, output: { "start_anchor": null }
- Use null only when you are absolutely certain the segment does not appear as a standalone section (e.g., no relevant heading and no identifiable transition).
- Do not invent or approximate an anchor for a missing segment.
- Even if a segment is null, the remaining non-null segment starts must still follow the strict forward-only order.

### Step 3.5: heuristics for identifying the case segments (context cues)
Use these context cues to identify where each segment starts. Prefer explicit headings when present, but headings are not required.

1. document_start:
  - Begins at the first non-whitespace character of the raw text
  - Often contains metadata, formation of the Court, parties/representatives, and sometimes a "Table of contents" and/or a "Summary"
  - Context cues to consider when locating the boundary:
    - "Judgment of the Court…"
    - "Case C-...", "Case T-...", or similar
    - "Table of contents", "Summary" (when these appear near the top before the numbered paragraphs)

2. procedural_introduction:
  - The procedural context of the case (how the case came before the Court / procedural history / what the request or action concerns)
  - A strong cue is that the first numbered paragraph often begins here
  - Context cues to consider when locating the boundary:
    - "Procedure before the Court"
    - "This request for a preliminary ruling concerns..."
    - Often preceded by formulas such as: "after hearing the Opinion of the Advocate General, gives the following"
  - Note: a top-of-document "Summary" usually belongs to document_start

3. legal_context:
  - The relevant legal framework.
  - Includes descriptions of EU law, national law, procedural rules, or applicable provisions that set the legal context for the analysis.
  - Typically appears before the factual narrative is discussed in detail and before the Court’s assessment
  - Context cues to consider when locating the boundary:
    - "Legal framework"
    - "Applicable law"
    - "EU law"
    - "National law"
    - "Relevant provisions"
 - If no standalone legal framework section can be identified when scanning forward, set start_anchor to null (per the null rule).

4. facts_and_background:
  - The factual narrative and background to the dispute.
  - Includes the events leading to the case, the dispute in the main proceedings, and/or relevant factual findings.
  - Context cues to consider when locating the boundary:
    - "Background to the dispute"
    - "Background"
    - "Facts"
    - "Facts in the main proceedings"
    - "The dispute in the main proceedings"
    
5. reasoning:
  - The Court’s analysis applying the law to the facts.
  - Includes assessment of pleas/grounds, answers to referred questions, interpretation, proportionality analysis, admissibility/substance, and the Court’s reasoning leading to its conclusion.
  - Context cues to consider when locating the boundary:
    - "Assessment by the Court"
    - "Grounds"
    - "Pleas in law"
    - "The first/second ground of appeal"
    - "Consideration of the questions referred"
    - "Substance"
    - "Admissibility"

6. decision_on_costs:
  - The part allocating costs (who bears costs, apportionment, interveners, etc.).
  - Usually fairly short and located under a dedicated “Costs” heading.
  - Context cues to consider when locating the boundary:
    - "Costs"

7. operative_part:
  - The dispositive section where the Court formally rules on the outcome.
  - Contains the final orders (e.g., answers to questions, dismissal/upholding of an action/appeal, annulment, declarations).
  - Context cues to consider when locating the boundary:
    - "On those grounds, the Court hereby rules:"
    - "For these reasons, the Court hereby:"
    - "The Court hereby declares and rules:"
    - Often formatted as numbered points/orders.

8. document_end:
  - Any formal closing statements after the operative part
  - Includes signatures, authenticity lines, language of the case, or closing formalities
  - Ends at the final character of the raw text.
  - Context cues to consider when locating the boundary:
    - Signatures, name blocks, registrar/president lines, "(Signatures)"
    - "Language of the case: ..."
  - If the operative part ends at the end of the raw text (that is, no closing statement follows), set start_anchor to null (per the null rule).

### Step 3.6: boundary selection rule (tie-breaker)
- If boundaries are ambiguous:
  - Prefer starting a segment at an explicit heading line (if present), so the heading appears in start_anchor.
  - Otherwise, start at the first line/paragraph that clearly belongs to that segment by content.
  - If selecting an anchor would violate the strict forward-only order, set that segment to null (only if you are certain it is not present as a standalone section).

---

## Validation checklist
- Each non-null start_anchor contains at most 30 words (or fewer if the segment has fewer words).
- document_start.start_anchor begins with the first word of the raw text.
- start_anchor words are copied verbatim from the raw text (no edits, no extra words, no ellipses).
- All non-null segment starts respect the strict forward-only order (no backtracking).
- Output only the JSON object required by the schema; do not output anything else.

---

## Final JSON schema

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CJEU case segmentation",
  "description": "Metadata, summary, and 8-part segmentation of CJEU case law",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "case_metadata": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "celex": { "type": "string" },
        "document_link": {
          "anyOf": [
            { "type": "string", "format": "uri" },
            { "const": "unknown" }
          ]
        },
        "ecli": { "type": "string" },
        "case_number": { "type": "string" },
        "date_of_judgment": {
          "anyOf": [
            { "type": "string", "format": "date" },
            { "const": "unknown" }
          ]
        },
        "document_type": { "type": "string" },
        "court": { "type": "string" },
        "case_name": { "type": "string" }
      },
      "required": [
        "celex",
        "document_link",
        "ecli",
        "case_number",
        "date_of_judgment",
        "document_type",
        "court",
        "case_name"
      ]
    },
    "case_summary": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "subject_matter": { "type": "string" },
        "keywords": {
          "type": "array",
          "minItems": 3,
          "maxItems": 5,
          "items": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "term": { "type": "string" },
              "description": { "type": "string" }
            },
            "required": ["term", "description"]
          }
        },
        "summary_text": { "type": "string" }
      },
      "required": ["subject_matter", "keywords", "summary_text"]
    },
    "case_segments": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "document_start": {
          "type": "array",
          "minItems": 1,
          "maxItems": 1,
          "items": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "start_anchor": { "anyOf": [{ "type": "string" }, { "type": "null" }] }
            },
            "required": ["start_anchor"]
          }
        },
        "procedural_introduction": {
          "type": "array",
          "minItems": 1,
          "maxItems": 1,
          "items": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "start_anchor": { "anyOf": [{ "type": "string" }, { "type": "null" }] }
            },
            "required": ["start_anchor"]
          }
        },
        "legal_context": {
          "type": "array",
          "minItems": 1,
          "maxItems": 1,
          "items": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "start_anchor": { "anyOf": [{ "type": "string" }, { "type": "null" }] }
            },
            "required": ["start_anchor"]
          }
        },
        "facts_and_background": {
          "type": "array",
          "minItems": 1,
          "maxItems": 1,
          "items": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "start_anchor": { "anyOf": [{ "type": "string" }, { "type": "null" }] }
            },
            "required": ["start_anchor"]
          }
        },
        "reasoning": {
          "type": "array",
          "minItems": 1,
          "maxItems": 1,
          "items": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "start_anchor": { "anyOf": [{ "type": "string" }, { "type": "null" }] }
            },
            "required": ["start_anchor"]
          }
        },
        "decision_on_costs": {
          "type": "array",
          "minItems": 1,
          "maxItems": 1,
          "items": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "start_anchor": { "anyOf": [{ "type": "string" }, { "type": "null" }] }
            },
            "required": ["start_anchor"]
          }
        },
        "operative_part": {
          "type": "array",
          "minItems": 1,
          "maxItems": 1,
          "items": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "start_anchor": { "anyOf": [{ "type": "string" }, { "type": "null" }] }
            },
            "required": ["start_anchor"]
          }
        },
        "document_end": {
          "type": "array",
          "minItems": 1,
          "maxItems": 1,
          "items": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "start_anchor": { "anyOf": [{ "type": "string" }, { "type": "null" }] }
            },
            "required": ["start_anchor"]
          }
        }
      },
      "required": [
        "document_start",
        "procedural_introduction",
        "legal_context",
        "facts_and_background",
        "reasoning",
        "decision_on_costs",
        "operative_part",
        "document_end"
      ]
    }
  },
  "required": ["case_metadata", "case_summary", "case_segments"]
}

---

## Few-Shot Examples (For guidance only. Do not parse this as the task input)

### Example 1: (illustrative)

#### Raw text of example 1
<<<BEGINNING OF RAW TEXT>>>
62021CJ0750\n\n \nJUDGMENT OF THE COURT (Second Chamber)\n\n8 February 2024 (\n\n*1\n\n)\n\n(Appeal – Economic and monetary policy – Prudential supervision of credit institutions – Tasks of the European Central Bank (ECB) – Regulation (EU) No 1024/2013 – Article 6(5)(b) – Supervision of a credit institution directly by the ECB – Conditions – Action for annulment – Inadmissibility – Representation of a party – Authority to act granted to the lawyer – Representative not lawfully provided with authority to act)\n\nIn Case C‐750/21 P,\n\nAPPEAL under Article 56 of the Statute of the Court of Justice of the European Union, brought on 6 December 2021,\n\nPilatus Bank plc\n, established in Ta’Xbiex (Malta), represented by O. Behrends, Rechtsanwalt,\n\nappellant,\n\nthe other party to the proceedings being:\n\nEuropean Central Bank (ECB)\n, represented by E. Koupepidou, M. Puidokas and E. Yoo, acting as Agents,\n\ndefendant at first instance,\n\nTHE COURT (Second Chamber),\n\ncomposed of A. Prechal, President of the Chamber, F. Biltgen, N. Wahl (Rapporteur), J. Passer and M.L. Arastey Sahún, Judges,\n\nAdvocate General: J. Kokott,\n\nRegistrar: A. Calot Escobar,\n\nhaving regard to the written procedure,\n\nafter hearing the Opinion of the Advocate General at the sitting on 25 May 2023,\n\ngives the following\n\nJudgment\n\n1\n\nBy its appeal, Pilatus Bank plc asks the Court of Justice to set aside the order of the General Court of the European Union of 24 September 2021, \nPilatus Bank\n v \nECB\n (T‐139/19, ‘the order under appeal’, \nEU:T:2021:623\n), by which the General Court dismissed as manifestly lacking any foundation in law Pilatus Bank’s action for annulment of the decision of the European Central Bank (ECB) of 21 December 2018 by which the ECB informed Pilatus Bank by email that the ECB was no longer competent to ensure its direct prudential supervision and to take measures concerning it (‘the email at issue’).\n\nLegal context\n\nRegulation (EU) No 1024/2013\n\n2\n\nArticle 1 of Council Regulation (EU) No 1024/2013 of 15 October 2013 conferring specific tasks on the European Central Bank concerning policies relating to the prudential supervision of credit institutions (\nOJ 2013 L 287, p. 63\n), entitled ‘Subject matter and scope’, states:\n\n‘This Regulation confers on the ECB specific tasks concerning policies relating to the prudential supervision of credit institutions, with a view to contributing to the safety and soundness of credit institutions and the stability of the financial system within the Union and each Member State, with full regard and duty of care for the unity and integrity of the internal market based on equal treatment of credit institutions with a view to preventing regulatory arbitrage.\n\n...’\n\n3\n\nArticle 2 of that regulation, entitled ‘Definitions’, provides, in paragraphs 2 and 3 thereof:\n\n‘For the purposes of this Regulation, the following definitions shall apply:\n\n...\n\n(2)   “national competent authority” means a national competent authority designated by a participating Member State in accordance with Regulation (EU) No 575/2013 of the European Parliament and of the Council of 26 June 2013 on prudential requirements for credit institutions and investment firms [and amending Regulation (EU) No 648/2012 (\nOJ 2013 L 176, p. 1\n)] and Directive 2013/36/EU [of the European Parliament and of the Council of 26 June 2013 on access to the activity of credit institutions and the prudential supervision of credit institutions and investment firms, amending Directive 2002/87/EC and repealing Directives 2006/48/EC and 2006/49/EC (\nOJ 2013 L 176, p. 338\n)];\n\n(3)   “credit institution” means a credit institution as defined in point 1 of Article 4(1) of [Regulation No 575/2013]’.\n\n4\n\nArticle 4 of that regulation defines the tasks conferred on the ECB and provides, in paragraph 1(a) and the first subparagraph of paragraph 3 thereof:\n\n‘1.   Within the framework of Article 6, the ECB shall, in accordance with paragraph 3 of this Article, be exclusively competent to carry out, for prudential supervisory purposes, the following tasks in relation to all credit institutions established in the participating Member States:\n\n(a)\n\nto authorise credit institutions and to withdraw authorisations of credit institutions subject to Article 14;\n\n...\n\n3.   For the purpose of carrying out the tasks conferred on it by this Regulation, and with the objective of ensuring high standards of supervision, the ECB shall apply all relevant Union law, and where this Union law is composed of Directives, the national legislation transposing those Directives. Where the relevant Union law is composed of Regulations and where currently those Regulations explicitly grant options for Member States, the ECB shall apply also the national legislation exercising those options.’\n\n5\n\nArticle 6 of that regulation, entitled ‘Cooperation within the [Single Supervisory Mechanism (SSM)]’, states:\n\n‘1.   The ECB shall carry out its tasks within a single supervisory mechanism composed of the ECB and national competent authorities. The ECB shall be responsible for the effective and consistent functioning of the SSM.\n\n2.   Both the ECB and national competent authorities shall be subject to a duty of cooperation in good faith, and an obligation to exchange information.\n\nWithout prejudice to the ECB’s power to receive directly, or have direct access to information reported, on an ongoing basis, by credit institutions, the national competent authorities shall in particular provide the ECB with all information necessary for the purposes of carrying out the tasks conferred on the ECB by this Regulation.\n\n3.   Where appropriate and without prejudice to the responsibility and accountability of the ECB for the tasks conferred on it by this Regulation, national competent authorities shall be responsible for assisting the ECB, under the conditions set out in the framework mentioned in paragraph 7 of this Article, with the preparation and implementation of any acts relating to the tasks referred to in Article 4 related to all credit institutions, including assistance in verification activities. They shall follow the instructions given by the ECB when performing the tasks mentioned in Article 4.\n\n4.   In relation to the tasks defined in Article 4 except for points (a) and (c) of paragraph 1 thereof, the ECB shall have the responsibilities set out in paragraph 5 of this Article and the national competent authorities shall have the responsibilities set out in paragraph 6 of this Article, within the framework and subject to the procedures referred to in paragraph 7 of this Article, for the supervision of the following credit institutions, financial holding companies or mixed financial holding companies, or branches, which are established in participating Member States, of credit institutions established in non-participating Member States:\n\n–\n\nthose that are less significant on a consolidated basis, at the highest level of consolidation within the participating Member States, or individually in the specific case of branches, which are established in participating Member States, of credit institutions established in non-participating Member States. The significance shall be assessed based on the following criteria:\n\n(i)\n\nsize;\n\n(ii)\n\nimportance for the economy of the Union or any participating Member State;\n\n(iii)\n\nsignificance of cross-border activities.\n\nWith respect to the first subparagraph above, a credit institution or financial holding company or mixed financial holding company shall not be considered less significant, unless justified by particular circumstances to be specified in the methodology, if any of the following conditions is met:\n\n(i)\n\nthe total value of its assets exceeds EUR 30 billion;\n\n(ii)\n\nthe ratio of its total assets over the [gross domestic product (GDP)] of the participating Member State of establishment exceeds 20%, unless the total value of its assets is below EUR 5 billion;\n\n(iii)\n\nfollowing a notification by its national competent authority that it considers such an institution of significant relevance with regard to the domestic economy, the ECB takes a decision confirming such significance following a comprehensive assessment by the ECB, including a balance-sheet assessment, of that credit institution.\n\nThe ECB may also, on its own initiative, consider an institution to be of significant relevance where it has established banking subsidiaries in more than one participating Member States and its cross-border assets or liabilities represent a significant part of its total assets or liabilities subject to the conditions laid down in the methodology.\n\nThose for which public financial assistance has been requested or received directly from the [European Financial Stability Facility (EFSF)] or the [European Stability Mechanism (ESM)] shall not be considered less significant.\n\nNotwithstanding the previous subparagraphs, the ECB shall carry out the tasks conferred on it by this Regulation in respect of the three most significant credit institutions in each of the participating Member States, unless justified by particular circumstances.\n\n5.   With regard to the credit institutions referred to in paragraph 4, and within the framework defined in paragraph 7:\n\n(a)\n\nthe ECB shall issue regulations, guidelines or general instructions to national competent authorities, according to which the tasks defined in Article 4 excluding points (a) and (c) of paragraph 1 thereof are performed and supervisory decisions are adopted by national competent authorities.\n\nSuch instructions may refer to the specific powers in Article 16(2) for groups or categories of credit institutions for the purposes of ensuring the consistency of supervisory outcomes within the SSM;\n\n(b)\n\nwhen necessary to ensure consistent application of high supervisory standards, the ECB may at any time, on its own initiative after consulting with national competent authorities or upon request by a national competent authority, decide to exercise directly itself all the relevant powers for one or more credit institutions referred to in paragraph 4, including in the case where financial assistance has been requested or received indirectly from the EFSF or the ESM;\n\n(c)\n\nthe ECB shall exercise oversight over the functioning of the system, based on the responsibilities and procedures set out in this Article, and in particular point (c) of paragraph 7;\n\n(d)\n\nthe ECB may at any time make use of the powers referred to in Articles 10 to 13;\n\n(e)\n\nthe ECB may also request, on an ad hoc or continuous basis, information from the national competent authorities on the performance of the tasks carried out by them under this Article.\n\n6.   Without prejudice to paragraph 5 of this Article, national competent authorities shall carry out and be responsible for the tasks referred to in points (b), (d) to (g) and (i) of Article 4(1) and adopting all relevant supervisory decisions with regard to the credit institutions referred to in the first subparagraph of paragraph 4 of this Article, within the framework and subject to the procedures referred to in paragraph 7 of this Article.\n\nWithout prejudice to Articles 10 to 13, the national competent authorities and national designated authorities shall maintain the powers, in accordance with national law, to obtain information from credit institutions, holding companies, mixed holding companies and undertakings included in the consolidated financial situation of a credit institution and to perform on site inspections at those credit institutions, holding companies, mixed holding companies and undertakings. The national competent authorities shall inform the ECB, in accordance with the framework set out in paragraph 7 of this Article, of the measures taken pursuant to this paragraph and closely coordinate those measures with the ECB.\n\nThe national competent authorities shall report to the ECB on a regular basis on the performance of the activities performed under this Article.\n\n...\n\n8.   Wherever the ECB is assisted by national competent authorities or national designated authorities for the purpose of exercising the tasks conferred on it by this Regulation, the ECB and the national competent authorities shall comply with the provisions set out in the relevant Union acts in relation to the allocation of responsibilities and cooperation between competent authorities from different Member States.’\n\nRegulation No 575/2013\n\n6\n\nPoint 1 of Article 4(1) of Regulation No 575/2013, in the version applicable to the dispute that gave rise to the appeal, was worded as follows:\n\n‘For the purposes of this Regulation, the following definitions shall apply:\n\n(1)\n\n“credit institution” means an undertaking the business of which is to take deposits or other repayable funds from the public and to grant credits for its own account’.\n\nBackground to the dispute\n\n7\n\nPilatus Bank is a credit institution established in Malta. As a ‘less significant’ credit institution, within the meaning of Article 6(4) of Regulation No 1024/2013, the appellant was subject to direct prudential supervision by the Malta Financial Services Authority (‘the MFSA’), the ‘national competent authority’, within the meaning of Article 2(2) of that regulation.\n\n8\n\nMr Ali Sadr Hasheminejad, a shareholder of the appellant who indirectly holds 100% of its capital and voting rights, was arrested in the United States on six charges relating to his alleged participation in a scheme in which approximately 115 million United States dollars (USD) (approximately EUR 108 million) in payments to finance a real estate project in Venezuela were embezzled for the benefit of Iranian individuals and undertakings.\n\n9\n\nFollowing Mr Sadr Hasheminejad’s indictment in the United States, the appellant received withdrawal requests totalling EUR 51.4 million worth of deposits, that is approximately 40% of the deposits on its balance sheet.\n\n10\n\nIn that context, the MFSA adopted three directives concerning the appellant.\n\n11\n\nOn 21 March 2018, it adopted, first, a directive regarding the removal or suspension of voting rights by which it ordered, inter alia, that Mr Sadr Hasheminejad be removed from his post of director of the appellant with immediate effect and from all his other decision-making roles within it, that he suspend the exercise of his voting rights and that he refrain from any legal or judicial representation of the appellant and, second, a directive regarding the moratorium, by which it ordered the appellant not to authorise any banking transactions, in particular withdrawals and deposits by shareholders and members of the appellant’s Board of Directors.\n\n12\n\nOn 22 March 2018, the MFSA adopted a directive appointing a competent person with authority, under the terms of that appointment, to ‘assume all the powers, functions and duties of the Bank in respect of all assets, whether exercisable by the Bank in general meeting or by the Board of Directors or by any other person, including the legal and judicial representation of the Bank to the exclusion of the Bank and any other person’ (‘the competent person’).\n\n13\n\nOn 29 June 2018, the MFSA proposed that the ECB withdraw the appellant’s authorisation to take up the business of a credit institution in accordance with Article 14(5) of Regulation No 1024/2013.\n\n14\n\nDuring the administrative procedure for withdrawal of authorisation, the appellant’s Board of Directors gave authority to act to a lawyer, who established contact with the ECB.\n\n15\n\nIn that context, the appellant, through the lawyer provided with authority to act by its Board of Directors, stated, in its correspondence with the ECB, that the competent person had informed that lawyer, in essence, that he would not authorise the payment of the lawyer’s fees out of the funds of the bank which he was responsible for administering.\n\n16\n\nThe appellant also stated that it had asked the MFSA to instruct the competent person to authorise the use of the bank’s funds for the payment of that lawyer’s fees but had not obtained a reply.\n\n17\n\nBy decision of 2 November 2018, the ECB withdrew the appellant’s authorisation on the basis of Article 4(1)(a) and Article 14(5) of Regulation No 1024/2013.\n\n18\n\nAgainst that background, the appellant asked the ECB, by two emails of 13 November and 20 December 2018, to ensure its direct prudential supervision under Regulation No 1024/2013 and to order the competent person to authorise the payment, out of the funds of the bank, of the fees of the lawyer provided with authority to act by its Board of Directors.\n\n19\n\nIn the email at issue, the ECB replied as follows:\n\n‘We refer to your email sent to the ECB on 13 November 2018 in which you request the ECB to take over direct supervision of [the appellant] and to comment on “the events of Friday 2 November 2018” as well as to your email of 20 December 2018 in which you reiterate your request for the take-over of the direct supervision of [the appellant] by the ECB. Please note, however, that the ECB’s supervisory tasks under [Regulation No 1024/2013] are limited to credit institutions (see [the first paragraph of] Article 1 ... of the said Regulation). Given that the authorisation of [the appellant] as a credit institution has been withdrawn with effect of 5 November 2018, the ECB is no longer competent to take any measures in respect of [the appellant].’\n\nThe action before the General Court and the order under appeal\n\n20\n\nBy application lodged at the Registry of the General Court on 4 March 2019, the appellant, through the lawyer provided with authority to act by its Board of Directors, brought an action for annulment of the email at issue.\n\n21\n\nBy the order under appeal, adopted on the basis of Article 126 of its Rules of Procedure, the General Court, without ruling on the plea of inadmissibility raised by the ECB, dismissed the action as manifestly lacking any foundation in law.\n\nThe procedure before the Court of Justice and the forms of order sought\n\n22\n\nBy document lodged at the Registry of the Court of Justice on 6 December 2021, the appellant, through the same lawyer as at first instance, brought the present appeal.\n\n23\n\nBy that appeal, it claims that the Court should:\n\n–\n\nset aside the order under appeal;\n\n–\n\ndeclare the email at issue void in accordance with Article 264 TFEU;\n\n–\n\nto the extent that the Court is not in a position to take a decision on the merits, refer the case back to the General Court for it to determine the action for annulment; and\n\n–\n\norder the ECB to pay all the costs.\n\n24\n\nThe ECB contends that the Court should:\n\n–\n\ndismiss the appeal as inadmissible in part and unfounded in part;\n\n–\n\nin the alternative, dismiss the appeal as unfounded in its entirety; and\n\n–\n\nin any event, order the appellant to pay all the costs.\n\nThe request to have the oral part of the procedure reopened\n\n25\n\nBy document lodged at the Court Registry on 27 June 2023, the ECB requested the Court to order the reopening of the oral part of the procedure, pursuant to Article 83 of the Rules of Procedure of the Court of Justice.\n\n26\n\nIn support of its request, the ECB states that it wishes to submit new facts which, in the light of recent events, namely the Advocate General’s Opinion of 25 May 2023, are of such a nature as to be a decisive factor for the decision of the Court. According to the ECB, it is apparent from the Opinion that the Advocate General considers that the directives concerning the appellant adopted by the MFSA in March 2018 are ‘preparatory acts’ in the composite administrative proceedings that led to the adoption by the ECB of the decision to withdraw authorisation and that the procedural deficiencies vitiating those directives are, therefore, attributable to the ECB and ‘contaminate’ the decision to withdraw authorisation adopted by the ECB. The ECB submits facts in order to show that those directives have been challenged before the Maltese courts.\n\n27\n\nIn that regard, it must be noted, first, that the Statute of the Court of Justice of the European Union and the Rules of Procedure of the Court make no provision for the interested parties referred to in Article 23 of the Statute to submit observations in response to the Advocate General’s Opinion (judgment of 9 June 2022, Préfet du Gers and Institut national de la statistique et des études économiques, \nC‐673/20\n, \nEU:C:2022:449\n, paragraph \n40\n and the case-law cited).\n\n28\n\nSecond, under the second paragraph of Article 252 TFEU, the Advocate General, acting with complete impartiality and independence, is to make, in open court, reasoned submissions on cases which, in accordance with the Statute of the Court of Justice of the European Union, require his or her involvement. It is not therefore an opinion addressed to the judges or to the parties which stems from an authority outside the Court, but rather, it is the individual reasoned opinion, expressed in open court, of a Member of the Court of Justice itself. In those circumstances, the Advocate General’s Opinion cannot be debated by the parties. Moreover, the Court is not bound either by the Advocate General’s submissions or by the reasoning which led to those submissions. Consequently, a party’s disagreement with the Opinion of the Advocate General, irrespective of the questions that he or she examines in the Opinion, cannot in itself constitute grounds justifying the reopening of the oral procedure (judgment of 9 June 2022, Préfet du Gers and Institut national de la statistique et des études économiques, \nC‐673/20\n, \nEU:C:2022:449\n, paragraph \n41\n and the case-law cited).\n\n29\n\nNevertheless, in accordance with Article 83 of its Rules of Procedure, the Court may at any time, after hearing the Advocate General, order the opening or reopening of the oral part of the procedure, in particular if it considers that it lacks sufficient information or where a party has, after the close of that part of the procedure, submitted a new fact which is of such a nature as to be a decisive factor for the decision of the Court.\n\n30\n\nHowever, in the present case, the Court considers that it has all the information necessary to give a ruling and that the evidence put forward by the ECB in support of its request that the oral part of the procedure be reopened does not constitute new facts of such a nature as to be a decisive factor for the decision of the Court which it is thus called upon to give.\n\n31\n\nIn those circumstances, the Court considers, after hearing the Advocate General, that there is no need to order the reopening of the oral part of the procedure.\n\nThe appeal\n\n32\n\nIt should be noted at the outset that, according to settled case-law, any fact which relates to the admissibility of the action for annulment brought before the General Court is likely to constitute a question of public policy which the Court of Justice, hearing an appeal, is required to raise of its own motion (judgments of 23 April 2009, Sahlstedt and Others v Commission, \nC‐362/06 P\n, \nEU:C:2009:243\n, paragraphs \n21\n to \n23\n, and of 6 July 2023, Julien v Council, \nC‐285/22 P\n, \nEU:C:2023:551\n, paragraph \n45\n and the case-law cited).\n\n33\n\nUnder Article 19 of the Statute of the Court of Justice of the European Union, which applies to the General Court pursuant to the first paragraph of Article 53 of that statute, in order to be able to bring proceedings before the Courts of the European Union, legal persons, such as the appellant, must be represented by a lawyer authorised to practise before a court of a Member State or of another State which is a party to the Agreement on the European Economic Area of 2 May 1992 (\nOJ 1994 L 1, p. 3\n).\n\n34\n\nThus, the representation of a legal person by a lawyer and, in particular, the question of the lawfulness of the authority granted to a lawyer for the purpose of bringing an action before the General Court are among the considerations of public policy which the Court of Justice, hearing an appeal, is required to raise of its own motion.\n\n35\n\nAs regards the authority to act conferred on a lawyer by such persons, Article 51(3) of the Rules of Procedure of the General Court provides that where the party represented by the lawyer is a legal person governed by private law, the lawyer must lodge at the Registry an authority to act given by that person. Unlike the version of those rules which applied before 1 July 2015, that provision does not require such a person to provide proof that the authority granted to its lawyer was conferred on him or her lawfully by someone authorised for that purpose.\n\n36\n\nHowever, the Court of Justice has already held that the fact that Article 51(3) does not lay down that obligation does not mean that the General Court need not verify whether the authority concerned is lawful where such authority is challenged. The fact that, at the stage of lodging its action, an applicant does not have to provide that proof does not affect the obligation on that party lawfully to have provided its lawyer with authority to act in order to be able to bring proceedings. The fact that the evidence requirements at the time of lodging an action have been relaxed has no bearing on the substantive condition that the parties must be duly represented by their lawyers. Accordingly, where the lawfulness of an authority granted by a party to its lawyer is challenged, that party must demonstrate that that authority is lawful (judgment of 21 September 2023, China Chamber of Commerce for Import and Export of Machinery and Electronic Products and Others v Commission, \nC‐478/21 P\n, \nEU:C:2023:685\n, paragraph \n93\n and the case-law cited).\n\n37\n\nThe General Court is also required to verify of its own motion that the authority to act in question is lawful and, in particular, that the authority to act has been conferred lawfully by a representative of the legal person in question who has the power to do so where such an authority to act is manifestly unlawful or where there are factors liable to cast serious doubt on the lawfulness of such an authority to act.\n\n38\n\nIn the present case, a number of circumstances should have led the General Court to have serious doubts as to the lawfulness of the authority to act given to the appellant’s lawyer.\n\n39\n\nThus, first, the factual circumstances that led to the action being brought before the General Court and the terms of the authority to act as a representative given by the appellant’s Board of Directors to the lawyer who brought that action were liable seriously to call into question the lawfulness of that authority to act.\n\n40\n\nThe appointment of the competent person by the MFSA and the fact that that competent person’s function was, inter alia, to assume ‘the legal and judicial representation of the Bank to the exclusion of the Bank and any other person’ were such as to give rise to serious doubts as to the capacity of the appellant’s Board of Directors to commit the appellant to legal proceedings and to provide a lawyer with authority to act for that purpose.\n\n41\n\nThe terms of the authority to act as a representative given to the lawyer were also liable to accentuate such doubts. Thus, the members of the appellant’s Board of Directors recalled, in that authority to act, that the MFSA had appointed the competent person on 22 March 2018 and that it had conferred certain powers on that person, stating: ‘Which persons are authorised to represent [the appellant] in the relevant context will have to be determined by the competent courts. The members of the [Board of Directors] do not assume any personal liability’. Those statements indicate that the signatories of the authority to act themselves had doubts as to their capacity to issue such an authority to act and constitute a clear and explicit invitation to verify that they actually had that capacity.\n\n42\n\nSecond, the action before the General Court sought the annulment of the ECB’s decision rejecting the appellant’s requests that the ECB ensure its direct prudential supervision and adopt various measures concerning it, by ordering, inter alia, the competent person to authorise the payment of the fees of the lawyer provided with authority to act by the appellant’s Board of Directors.\n\n43\n\nSuch requests made by the appellant to the ECB were also liable to cast serious doubt on the lawfulness of the authority of the appellant’s lawyer to act as a representative in the proceedings before the General Court. The fact that the fees of the appellant’s lawyer could not be paid was such as to indicate that the body that had provided the lawyer with authority to act had no power to make that payment and that it did not have the power to bring legal proceedings and to provide a lawyer with authority to act for that purpose.\n\n44\n\nThird, before the General Court, the appellant expressly claimed that the email at issue deprived it of the possibility of effective representation.\n\n45\n\nIn those circumstances, irrespective of the merits of that argument, the General Court had to require, of its own motion, proof that the lawyer representing the appellant had been lawfully provided with authority to act and that the authority to act had been conferred by someone authorised for that purpose.\n\n46\n\nIt follows from the foregoing considerations that the General Court erred in law by failing to determine of its own motion whether the authority conferred by the appellant’s Board of Directors on its lawyer was lawful.\n\n47\n\nSuch a manifest error must result in the order under appeal being set aside, without there being any need to rule on the grounds of appeal put forward by the appellant.\n\n48\n\nIn accordance with the first paragraph of Article 61 of the Statute of the Court of Justice of the European Union, if the Court sets aside the decision of the General Court, it may itself give final judgment in the matter, where the state of the proceedings so permits.\n\n49\n\nThat is the position in the present case. Since the parties were invited by the Court to comment on the admissibility of the action before the General Court and, in particular, on the lawfulness of the authority to act as a representative issued by the appellant’s Board of Directors, the Court has all the information necessary to rule on the admissibility of the action.\n\n50\n\nThe appellant claimed, on the basis of the judgment of the Qorti tal-Appell (Kompetenza Inferjuri) (Court of Appeal (Inferior Jurisdiction), Malta) of 5 November 2018 in Case No 6/2017 (\nHeikki Niemelä and Others\n v \nMaltese Financial Services Authority\n), that, despite the appointment of the competent person, its Board of Directors still had the power to represent it in legal proceedings and, to that end, to provide a lawyer with authority to act.\n\n51\n\nThus, according to the appellant, the only effect of appointing the competent person was to entrust the bank’s assets and the management of the bank’s business to that person, without, however, granting him the capacity to represent that bank in judicial proceedings challenging decisions that are binding on the bank. In that regard, it is irrelevant that such decisions might also affect the assets and business that are managed by the competent person.\n\n52\n\nThe appellant also emphasised that the judgment of 5 November 2019, \nECB and Others\n v \nTrasta Komercbanka and Others\n (C‐663/17 P, C‐665/17 P and C‐669/17 P, ‘the judgment in \nTrasta Komercbanka\n’, \nEU:C:2019:923\n), and the Opinion of the Advocate General relating to that judgment confirm that the issue of representation is determined primarily by national law and that the General Court’s finding in that regard is binding unless a party demonstrates that it constitutes a distortion of the facts. Under Maltese law, the representation of the bank does not fall within the remit of the competent person even if that person is in charge of the bank’s business or its assets.\n\n53\n\nThe ECB stated that the representation of a corporate legal person is governed by the \nlex incorporationis\n and that, in the present case, Maltese law, as interpreted by the judgment of the Qorti tal-Appell (Kompetenza Inferjuri) (Court of Appeal (Inferior Jurisdiction)) of 5 November 2018 in Case No 6/2017 (\nHeikki Niemelä and Others\n v \nMaltese Financial Services Authority\n), limits the competent person’s power to represent the appellant to the specific circumstances referred to in the national law on the basis of which the competent person was appointed, in particular as regards matters concerning the assets and exercising control of the business, and, therefore, the Board of Directors maintains residual rights.\n\n54\n\nIt also observed that the authority to act issued by the appellant’s Board of Directors covered only the representation in regulatory matters without explicitly mentioning judicial representation.\n\n55\n\nIn that regard, as the Court has already noted in paragraph 33 of the present judgment, under Article 19 of the Statute of the Court of Justice of the European Union, which applies to the General Court pursuant to the first paragraph of Article 53 of that statute, in order to be able to bring proceedings before the Courts of the European Union, legal persons, such as the appellant, must be represented by a lawyer authorised to practise before a court of a Member State or of another State which is a party to the Agreement on the European Economic Area.\n\n56\n\nIn view of that need for legal persons to be represented by a lawyer authorised to practise before a court of a Member State or of another State which is a party to the Agreement on the European Economic Area, the admissibility of an action for annulment brought by such a person under Article 263 TFEU is subject to proof that the person concerned has indeed made the decision to bring the action and that the lawyers who claim to represent that person have in fact been authorised to do so (see, to that effect, judgment in \nTrasta Komercbanka\n, paragraph 57 and the case-law cited).\n\n57\n\nIt is precisely in order to ensure that that is indeed the case that Article 51(3) of the Rules of Procedure of the General Court requires lawyers, where the party they represent is a legal person governed by private law, to lodge at the Registry of the General Court an authority to act given by that person, as failure to produce that authority to act may entail, in accordance with Article 51(4) of those rules, the formal inadmissibility of the application (judgment in \nTrasta Komercbanka\n, paragraph 57).\n\n58\n\nIn the case of a credit institution constituted in the form of a legal person governed by the law of a Member State, such as the appellant, where there are no EU rules in the matter, it is under that law that it is necessary to determine which bodies of that legal person are entitled to take the decisions referred to in paragraphs 56 and 57 of the present judgment (judgment in \nTrasta Komercbanka\n, paragraph 58).\n\n59\n\nIn the present case, it must be held that, having regard to the authority to act granted to the competent person and, in particular, to the fact that it was for that person to ‘assume all the powers, functions and duties of the Bank in respect of all assets, whether exercisable by the Bank in general meeting or by the Board of Directors or by any other person, including the legal and judicial representation of the Bank to the exclusion of the Bank and any other person’, the appellant’s Board of Directors was no longer entitled to represent the appellant and no longer had the power to provide a lawyer with authority to act for that purpose.\n\n60\n\nThe power of the appellant’s Board of Directors to represent it in legal proceedings and to provide a lawyer with authority to act for that purpose cannot, moreover, be based on the judgment in \nTrasta Komercbanka\n.\n\n61\n\nThat judgment concerns the obligation of a Court of the European Union not to take account of the revocation of the authority conferred on a party’s representative where that revocation infringes that party’s right to effective judicial protection. However, such an obligation is incumbent on a Court of the European Union only in certain defined circumstances.\n\n62\n\nAs is apparent from paragraphs 60 to 62 of the judgment in \nTrasta Komercbanka\n, the Court held that the right of the credit institution Trasta Komercbanka to an effective remedy was infringed on account of the fact that the liquidator appointed after authorisation was withdrawn and after that institution was placed in liquidation had a conflict of interests. The Court noted that the liquidator, whose task was to carry out the final liquidation of that institution, had been appointed on a proposal from the national competent authority, which could at any time relieve the liquidator of its duties. It considered, consequently, that there was a risk that that liquidator might avoid challenging, in court proceedings, the decision to withdraw that institution’s authorisation, which had been adopted by the ECB on a proposal from that authority and which had led to that institution being placed in liquidation. The Court concluded, in paragraph 78 of that judgment, that the revocation, by the liquidator, of the authority to act conferred by the former decision-making bodies of Trasta Komercbanka on the lawyer who had brought an action against that decision infringed that institution’s right to effective judicial protection and that, in taking that revocation into account, the General Court had erred in law.\n\n63\n\nIn the present case, the authority to act of the competent person appointed by the MFSA differs significantly from that of the liquidator as described in paragraph 72 of the judgment in \nTrasta Komercbanka\n, since the latter’s sole purpose was to collect debts, sell assets and satisfy claims of creditors in order to bring about the total cessation of the activity of the credit institution concerned.\n\n64\n\nIn addition, the appellant has not adduced any evidence relating to the competent person’s authority to act or the conditions under which he exercises that authority indicating that that person had, in law or in fact, a conflict of interests. In particular, it in no way follows from the wording of that authority to act, referred to in paragraph 59 of the present judgment, that the competent person does not represent the interests of the bank.\n\n65\n\nSimilarly, the fact that the competent person was appointed by the national competent authority that submitted to the ECB the proposal to withdraw authorisation is not sufficient, in itself, to establish the existence of a conflict of interests.\n\n66\n\nAs regards the scope of the judgment referred to in paragraph 50 of the present judgment, it should be noted that it did not concern the appellant but another Maltese credit institution in respect of which the MFSA had appointed a competent person.\n\n67\n\nMoreover, the Qorti tal-Appell (Kompetenza Inferjuri) (Court of Appeal (Inferior Jurisdiction)) confirmed, in that judgment, that the directors of a credit institution are not divested of all their powers by virtue of the appointment of a competent person. They are, thus, still empowered to request, on behalf of the credit institution, the revocation of a number of prudential supervisory decisions issued by the MFSA as the national competent authority and, in particular, of the decision appointing a competent person.\n\n68\n\nHowever, it does not follow from that judgment that, where a competent person has been appointed and authority to act as a representative, in particular a judicial representative, has been conferred on that person, the directors of a credit institution retain the power to provide a lawyer with authority to act as a representative of that institution in proceedings concerning decisions adopted by the ECB or to challenge the decisions of that institution.\n\n69\n\nLast, it is irrelevant that the appellant’s Board of Directors is the addressee of the email at issue in so far as that email was adopted by the ECB in response to a request made by the lawyer provided with authority to act by that body.\n\n70\n\nWhile it may follow from such a circumstance that the appellant’s Board of Directors, as the addressee of the email at issue, would have standing to bring an action for the annulment of that email in its own name, that does not mean that, following the appointment of the competent person, the Board of Directors was still entitled to take the decision to bring an action before a Court of the European Union on behalf of the appellant and still had the power to provide a lawyer with authority to act for that purpose.\n\n71\n\nIn the light of all the foregoing considerations, the action at first instance must be dismissed as inadmissible.\n\nCosts\n\n72\n\nUnder Article 184(2) of the Rules of Procedure of the Court, where an appeal is well founded and the Court itself gives final judgment in the case, the Court is to make a decision as to costs.\n\n73\n\nUnder Article 138(1) of those rules, applicable to the procedure on an appeal by virtue of Article 184(1) thereof, the unsuccessful party is to be ordered to pay the costs if they have been applied for in the successful party’s pleadings.\n\n74\n\nIn the present case, since Pilatus Bank has been unsuccessful and the ECB has applied, both before the Court of Justice and before the General Court, for Pilatus Bank to be ordered to pay the costs, Pilatus Bank must be ordered to bear its own costs and to pay those incurred by the ECB in the proceedings at first instance and in the present appeal.\n\n \n\nOn those grounds, the Court (Second Chamber) hereby:\n\n \n\n1.\n\nSets aside the order of the General Court of the European Union of 24 September 2021, Pilatus Bank v ECB (\nT‐139/19\n, \nEU:T:2021:623\n);\n\n \n\n2.\n\nDismisses the action brought in Case T‐139/19 as inadmissible;\n\n \n\n3.\n\nOrders Pilatus Bank plc to pay the costs.\n\n \n\nPrechal\n\nBiltgen\n\nWahl\n\nPasser\n\nArastey Sahún\n\nDelivered in open court in Luxembourg on 8 February 2024.\n\nA. Calot Escobar\n\nRegistrar\n\nA. Prechal\n\nPresident of the Chamber\n\n(\n\n*1\n\n)\tLanguage of the case: English.
<<<END OF RAW TEXT>>>

#### Filled-out final JSON schema of example 1
{
  "case_metadata": {
    "celex": "62021CJ0750",
    "document_link": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=cellar:8677f8ed-c665-11ee-95d9-01aa75ed71a1",
    "ecli": "ECLI:EU:C:2024:124",
    "case_number": "C-750/21 P",
    "date_of_judgment": "2024-02-08",
    "document_type": "Judgment",
    "court": "Court of Justice",
    "case_name": "Pilatus Bank plc v European Central Bank"
  },
  "case_summary": {
    "subject_matter": "Appeal concerning the ECB's prudential supervision competence over a credit institution after its authorisation has been withdrawn and the lawfulness of legal representation for such an institution.",
    "keywords": [
      {
        "term": "Prudential supervision",
        "description": "The monitoring of credit institutions by the ECB and national authorities to ensure financial stability and soundness."
      },
      {
        "term": "Legal representation",
        "description": "The requirement for legal persons to be represented by a lawyer and the necessity of a valid authority to act."
      },
      {
        "term": "Withdrawal of authorisation",
        "description": "The formal removal of a credit institution's license to operate, affecting the ECB's ongoing supervisory competence."
      },
      {
        "term": "Effective judicial protection",
        "description": "The right of a party to have access to a court and be represented, particularly in cases of potential conflicts of interest."
      }
    ],
    "summary_text": "Pilatus Bank plc appealed an order of the General Court which dismissed its action for annulment against an ECB email stating the ECB was no longer competent for its supervision following the withdrawal of its banking license. The General Court had dismissed the action as manifestly lacking foundation. On appeal, the Court of Justice raised a public policy issue regarding the bank's legal representation. It noted that the Maltese Financial Services Authority had appointed a 'competent person' to assume all powers of the bank, including legal representation, to the exclusion of the Board of Directors. The Court found that the General Court erred in law by failing to verify of its own motion whether the lawyer's authority to act, granted by the Board of Directors rather than the competent person, was lawful. Setting aside the order, the Court of Justice gave final judgment on the admissibility. It determined that under Maltese law, the Board of Directors no longer had the power to represent the bank or authorize a lawyer once the competent person was appointed. Unlike the Trasta Komercbanka case, no conflict of interest was proven to exist with the competent person that would necessitate allowing the Board to act to ensure effective judicial protection. Consequently, the original action was declared inadmissible because the lawyer lacked a valid authority to act from a body empowered to represent the bank."
  },
  "case_segments": {
    "document_start": [
      {
        "start_anchor": "62021CJ0750\n\n \nJUDGMENT OF THE COURT (Second Chamber)\n\n8 February 2024 (\n\n*1\n\n)\n\n(Appeal – Economic and monetary policy – Prudential supervision of credit institutions – Tasks of the European Central Bank (ECB) – Regulation"
      }
    ],
    "procedural_introduction": [
      {
        "start_anchor": "Judgment\n\n1\n\nBy its appeal, Pilatus Bank plc asks the Court of Justice to set aside the order of the General Court of the European Union of 24 September 2021,"
      }
    ],
    "legal_context": [
      {
        "start_anchor": "Legal context\n\nRegulation (EU) No 1024/2013\n\n2\n\nArticle 1 of Council Regulation (EU) No 1024/2013 of 15 October 2013 conferring specific tasks on the European Central Bank concerning policies relating to"
      }
    ],
    "facts_and_background": [
      {
        "start_anchor": "Background to the dispute\n\n7\n\nPilatus Bank is a credit institution established in Malta. As a ‘less significant’ credit institution, within the meaning of Article 6(4) of Regulation No 1024/2013,"
      }
    ],
    "reasoning": [
      {
        "start_anchor": "The appeal\n\n32\n\nIt should be noted at the outset that, according to settled case-law, any fact which relates to the admissibility of the action for annulment brought before the"
      }
    ],
    "decision_on_costs": [
      {
        "start_anchor": "Costs\n\n72\n\nUnder Article 184(2) of the Rules of Procedure of the Court, where an appeal is well founded and the Court itself gives final judgment in the case, the"
      }
    ],
    "operative_part": [
      {
        "start_anchor": "On those grounds, the Court (Second Chamber) hereby:\n\n \n\n1.\n\nSets aside the order of the General Court of the European Union of 24 September 2021, Pilatus Bank v ECB"
      }
    ],
    "document_end": [
      {
        "start_anchor": "Prechal\n\nBiltgen\n\nWahl\n\nPasser\n\nArastey Sahún\n\nDelivered in open court in Luxembourg on 8 February 2024.\n\nA. Calot Escobar\n\nRegistrar\n\nA. Prechal\n\nPresident of the Chamber\n\n(\n\n*1\n\n)\tLanguage of the case: English."
      }
    ]
  }
}

### Example 2: (illustrative)

#### Raw text of example 2
<<<BEGINNING OF RAW TEXT>>>
62023CJ0035\n\n \nJUDGMENT OF THE COURT (Fourth Chamber)\n\n20 June 2024 (\n\n*1\n\n)\n\n(Reference for a preliminary ruling – Judicial cooperation in civil matters – Parental responsibility – Regulation (EC) No 2201/2003 – Articles 10 and 11 – Jurisdiction in cases of the wrongful removal of a child – Child’s habitual residence in a Member State before the wrongful removal – Return procedure between a third country and a Member State – Concept of ‘request for return’ – The Hague Convention of 25 October 1980 on the Civil Aspects of International Child Abduction)\n\nIn Case C‐35/23 [Greislzel], (\n\n1\n\n)\n\nREQUEST for a preliminary ruling under Article 267 TFEU from the Oberlandesgericht Frankfurt am Main (Higher Regional Court, Frankfurt am Main, Germany), made by decision of 16 January 2023, received at the Court on 25 January 2023, in the proceedings\n\nFather\n\nv\n\nMother,\n\nOther parties to the proceedings:\n\nChild L,\n\nLawyer,\n\nTHE COURT (Fourth Chamber),\n\ncomposed of C. Lycourgos, President of the Chamber, O. Spineanu‐Matei, J.‐C. Bonichot, S. Rodin and L.S. Rossi (Rapporteur), Judges,\n\nAdvocate General: M. Campos Sánchez-Bordona,\n\nRegistrar: N. Mundhenke, Administrator,\n\nhaving regard to the written procedure and further to the hearing on 7 December 2023,\n\nafter considering the observations submitted on behalf of:\n\n–\n\nthe father, by A. Hamerak and T. von Plehwe, Rechtsanwälte,\n\n–\n\nthe German Government, by J. Möller, M. Hellmann, R. Kanitz and J. Simon, acting as Agents,\n\n–\n\nthe Polish Government, by B. Majczyna, M. Kozak and S. Żyrek, acting as Agents,\n\n–\n\nthe European Commission, by C. Vollrath and W. Wils, acting as Agents,\n\nafter hearing the Opinion of the Advocate General at the sitting on 8 February 2024,\n\ngives the following\n\nJudgment\n\n1\n\nThis request for a preliminary ruling concerns the interpretation of Articles 10 and 11 of Council Regulation (EC) No 2201/2003 of 27 November 2003 concerning jurisdiction and the recognition and enforcement of judgments in matrimonial matters and the matters of parental responsibility, repealing Regulation (EC) No 1347/2000 (\nOJ 2003 L 338, p. 1\n).\n\n2\n\nThe request has been made in proceedings between a German national residing in Switzerland, the father of L, a minor child, and the child’s mother, concerning parental responsibility over that child.\n\nLegal context\n\nThe 1980 Hague Convention\n\n3\n\nAs set out in the preamble to the Convention on the Civil Aspects of International Child Abduction, concluded in The Hague on 25 October 1980 (‘the 1980 Hague Convention’), that convention seeks to ‘protect children internationally from the harmful effects of their wrongful removal or retention and to establish procedures to ensure their prompt return to the State of their habitual residence, as well as to secure protection for rights of access’.\n\n4\n\nThe first paragraph of Article 6 of that convention provides:\n\n‘A Contracting State shall designate a Central Authority to discharge the duties which are imposed by the [c]onvention upon such authorities.’\n\n5\n\nThe first paragraph of Article 8 of that convention states:\n\n‘Any person, institution or other body claiming that a child has been removed or retained in breach of custody rights may apply either to the Central Authority of the child’s habitual residence or to the Central Authority of any other Contracting State for assistance in securing the return of the child.’\n\n6\n\nAccording to the first paragraph of Article 12 of the convention:\n\n‘Where a child has been wrongfully removed or retained in terms of Article 3 and, at the date of the commencement of the proceedings before the judicial or administrative authority of the Contracting State where the child is, a period of less than one year has elapsed from the date of the wrongful removal or retention, the authority concerned shall order the return of the child forthwith.’\n\n7\n\nArticle 13 of the 1980 Hague Convention provides:\n\n‘Notwithstanding the provisions of the preceding [a]rticle, the judicial or administrative authority of the requested State is not bound to order the return of the child if the person, institution or other body which opposes its return establishes that –\n\n(a)\n\nthe person, institution or other body having the care of the person of the child was not actually exercising the custody rights at the time of removal or retention, or had consented to or subsequently acquiesced in the removal or retention; or\n\n(b)\n\nthere is a grave risk that his or her return would expose the child to physical or psychological harm or otherwise place the child in an intolerable situation.\n\n...’\n\n8\n\nArticle 34 of the 1980 Hague Convention provides:\n\n‘... The present [c]onvention shall not restrict the application of an international instrument in force between the State of origin and the State addressed ... for the purposes of obtaining the return of a child who has been wrongfully removed or retained or of organising access rights.’\n\nRegulation No 2201/2003\n\n9\n\nRecitals 12, 17 and 18 of Regulation No 2201/2003 are worded as follows:\n\n‘(12)\n\nThe grounds of jurisdiction in matters of parental responsibility established in the present Regulation are shaped in the light of the best interests of the child, in particular on the criterion of proximity. This means that jurisdiction should lie in the first place with the Member State of the child’s habitual residence, except for certain cases of a change in the child’s residence or pursuant to an agreement between the holders of parental responsibility.\n\n...\n\n(17)\n\nIn cases of wrongful removal or retention of a child, the return of the child should be obtained without delay, and to this end the [1980 Hague Convention] would continue to apply as complemented by the provisions of this Regulation, in particular Article 11. The courts of the Member State to or in which the child has been wrongfully removed or retained should be able to oppose his or her return in specific, duly justified cases. However, such a decision could be replaced by a subsequent decision by the court of the Member State of habitual residence of the child prior to the wrongful removal or retention. Should that judgment entail the return of the child, the return should take place without any special procedure being required for recognition and enforcement of that judgment in the Member State to or in which the child has been removed or retained.\n\n(18)\n\nWhere a court has decided not to return a child on the basis of Article 13 of the 1980 Hague Convention, it should inform the court having jurisdiction or central authority in the Member State where the child was habitually resident prior to the wrongful removal or retention. Unless the court in the latter Member State has been seised, this court or the central authority should notify the parties. This obligation should not prevent the central authority from also notifying the relevant public authorities in accordance with national law.’\n\n10\n\nArticle 2 of that regulation, entitled ‘Definitions’, provides:\n\n‘For the purposes of this Regulation:\n\n...\n\n7.\n\nthe term “parental responsibility” shall mean all rights and duties relating to the person or the property of a child which are given to a natural or legal person by judgment, by operation of law or by an agreement having legal effect. The term shall include rights of custody and rights of access;\n\n...\n\n9.\n\nthe term “rights of custody” shall include rights and duties relating to the care of the person of a child, and in particular the right to determine the child’s place of residence;\n\n...\n\n11.\n\nthe term “wrongful removal or retention” shall mean a child’s removal or retention where:\n\n(a)\n\nit is in breach of rights of custody acquired by judgment or by operation of law or by an agreement having legal effect under the law of the Member State where the child was habitually resident immediately before the removal or retention;\n\nand\n\n(b)\n\nprovided that, at the time of removal or retention, the rights of custody were actually exercised, either jointly or alone, or would have been so exercised but for the removal or retention. Custody shall be considered to be exercised jointly when, pursuant to a judgment or by operation of law, one holder of parental responsibility cannot decide on the child's place of residence without the consent of another holder of parental responsibility.’\n\n11\n\nRegulation No 2201/2003 includes a Chapter II, entitled ‘Jurisdiction’, which contains, in Section 2, which is headed ‘Parental responsibility’, Articles 8 to 15 of that regulation.\n\n12\n\nArticle 8 of that regulation, headed ‘General jurisdiction’, provides:\n\n‘1.   The courts of a Member State shall have jurisdiction in matters of parental responsibility over a child who is habitually resident in that Member State at the time the court is seised.\n\n2.   Paragraph 1 shall be subject to the provisions of Articles 9, 10 and 12.’\n\n13\n\nArticle 10 of that regulation, headed ‘Jurisdiction in cases of child abduction’, provides:\n\n‘In case of wrongful removal or retention of the child, the courts of the Member State where the child was habitually resident immediately before the wrongful removal or retention shall retain their jurisdiction until the child has acquired a habitual residence in another Member State and:\n\n(a)\n\neach person, institution or other body having rights of custody has acquiesced in the removal or retention;\n\nor\n\n(b)\n\nthe child has resided in that other Member State for a period of at least one year after the person, institution or other body having rights of custody has had or should have had knowledge of the whereabouts of the child and the child is settled in his or her new environment and at least one of the following conditions is met:\n\n(i)\n\nwithin one year after the holder of rights of custody has had or should have had knowledge of the whereabouts of the child, no request for return has been lodged before the competent authorities of the Member State where the child has been removed or is being retained;\n\n(ii)\n\na request for return lodged by the holder of rights of custody has been withdrawn and no new request has been lodged within the time limit set in paragraph (i);\n\n(iii)\n\na case before the court in the Member State where the child was habitually resident immediately before the wrongful removal or retention has been closed pursuant to Article 11(7);\n\n(iv)\n\na judgment on custody that does not entail the return of the child has been issued by the courts of the Member State where the child was habitually resident immediately before the wrongful removal or retention.’\n\n14\n\nArticle 11 of Regulation No 2201/2003, headed ‘Return of the child’, provides:\n\n‘1.   Where a person, institution or other body having rights of custody applies to the competent authorities in a Member State to deliver a judgment on the basis of the [1980 Hague Convention], in order to obtain the return of a child that has been wrongfully removed or retained in a Member State other than the Member State where the child was habitually resident immediately before the wrongful removal or retention, paragraphs 2 to 8 shall apply.\n\n...\n\n6.   If a court has issued an order on non-return pursuant to Article 13 of the 1980 Hague Convention, the court must immediately either directly or through its central authority, transmit a copy of the court order on non-return and of the relevant documents, in particular a transcript of the hearings before the court, to the court with jurisdiction or central authority in the Member State where the child was habitually resident immediately before the wrongful removal or retention, as determined by national law. The court shall receive all the mentioned documents within one month of the date of the non-return order.\n\n7.   Unless the courts in the Member State where the child was habitually resident immediately before the wrongful removal or retention have already been seised by one of the parties, the court or central authority that receives the information mentioned in paragraph 6 must notify it to the parties and invite them to make submissions to the court, in accordance with national law, within three months of the date of notification so that the court can examine the question of custody of the child.\n\nWithout prejudice to the rules on jurisdiction contained in this Regulation, the court shall close the case if no submissions have been received by the court within the time limit.\n\n8.   Notwithstanding a judgment of non-return pursuant to Article 13 of the 1980 Hague Convention, any subsequent judgment which requires the return of the child issued by a court having jurisdiction under this Regulation shall be enforceable in accordance with Section 4 of Chapter III below in order to secure the return of the child.’\n\n15\n\nArticle 60 of that regulation, headed ‘Relations with certain multilateral conventions’, provides:\n\n‘In relations between Member States, this Regulation shall take precedence over the following [c]onventions in so far as they concern matters governed by this Regulation:\n\n...\n\n(e) the [1980 Hague Convention] ...’.\n\nThe dispute in the main proceedings and the questions referred for a preliminary ruling\n\n16\n\nL was born in Switzerland in November 2014 and has dual German and Polish nationality. Her father, a German national, has resided in Switzerland since June 2013 for professional reasons, while her mother, a Polish national, lived with her daughter from January 2015 to April 2016 in Frankfurt am Main (Germany), a city in which L’s parents married.\n\n17\n\nFrom January 2015 to April 2016, the father regularly visited the mother and L in Germany.\n\n18\n\nIn May 2015, the Swiss Migration Office accepted the application for family reunification made by the father, following which the mother obtained a temporary residence permit in Switzerland, valid until 31 December 2019.\n\n19\n\nOn 9 April 2016, the mother and L moved to Poland. The mother then made a statement of departure for the whole family in Frankfurt am Main, indicating the father’s address in Switzerland. In the summer of 2016, the mother applied for jobs in Switzerland. She has worked in Poland since November 2016.\n\n20\n\nInitially, the father visited his wife and daughter in Poland. However, from April 2017, the mother refused to allow the father his rights of access to their daughter. She enrolled her in a kindergarten in Poland, without the father’s consent. In late May 2017, the mother informed the father that she intended to remain in Poland with their daughter.\n\n21\n\nOn 7 July 2017, the father lodged an application for the return of the child to Switzerland through the Swiss Central Authority, namely the Federal Office of Justice in Berne, under the 1980 Hague Convention.\n\n22\n\nBy decision of 8 December 2017, the Sąd Rejonowy dla Krakowa-Nowej Huty w Krakowie (District Court of Krakow-Nowa Huta in Krakow, Poland) dismissed that application on the ground that the father had given his consent for an indefinite period to the mother’s move with their daughter to Poland. In addition, that court held that there was a grave risk to the best interests of the child, within the meaning of point (b) of the first paragraph of Article 13 of the 1980 Hague Convention, in the event of her return as the father had admitted to having once used violence against the mother.\n\n23\n\nThe appeal brought by the father against that decision was dismissed by the Sąd Okręgowy w Krakowie (Regional Court, Krakow, Poland) by decision of 17 April 2018.\n\n24\n\nOn 27 September 2017, the mother initiated divorce proceedings in Poland. In October 2017, she also made a statement of departure in respect of L to the municipality X in Switzerland.\n\n25\n\nBy decision of 5 June 2018, the Sąd Okręgowy w Krakowie (Regional Court, Krakow) provisionally granted the mother custody of L and set the father’s maintenance obligations. The referring court states that, in 2022, the father visited the child in Poland pursuant to a judicial decision taken in that Member State.\n\n26\n\nThe father did not pursue a second application for the return of the child under the 1980 Hague Convention, which he had lodged on 29 June 2018 with the Bundesamt für Justiz (Federal Office of Justice) in Bonn (Germany).\n\n27\n\nBy application of 12 July 2018, lodged with the Amtsgericht Frankfurt am Main (Local Court, Frankfurt am Main, Germany), the father applied for sole custody of the child, the right to determine the child’s residence and the return of the child to his home in Switzerland, as soon as the judgment entered into force.\n\n28\n\nThe father submitted that the child’s parents had agreed, during 2015, that they would continue to live with L in Switzerland in the future. In April 2016, the mother decided to join her parents, for a limited period, in Poland. The father agreed to it, provided that that stay was limited to two or three years. It was agreed that the child should attend a kindergarten in Switzerland from November 2017 at the latest.\n\n29\n\nThe mother opposed the application. She claims that the father consented to the move to Poland and provided assistance in the process of obtaining a Polish passport in that country. There had been, by contrast, no agreement on a time-limited move to Poland, just as there had been no agreement on a move to Switzerland.\n\n30\n\nBy decision of 3 June 2019, the Amtsgericht Frankfurt am Main (Local Court, Frankfurt am Main) dismissed the father’s application for the award of sole custody of the child, on the ground that that court did not have international jurisdiction to rule on the application.\n\n31\n\nThe father brought an appeal against that decision before the Oberlandesgericht Frankfurt am Main (Higher Regional Court, Frankfurt am Main, Germany) claiming, in essence, that the jurisdiction of the German courts derives from Article 11(6) of Regulation No 2201/2003, read in conjunction with paragraph 7 of that article, and from Article 10 of that regulation.\n\n32\n\nIn that regard, the referring court notes, in the first place, that, on the date on which the father’s application at first instance was lodged, namely 12 July 2018, L was habitually resident in Poland, with the result that the jurisdiction of the German courts cannot be based on Article 8(1) of Regulation No 2201/2003.\n\n33\n\nIn the second place, as regards the jurisdiction of those courts which, according to the father, derives from Articles 10 and 11 of Regulation No 2201/2003, the referring court takes the view that those articles must be interpreted together and points out that those articles apply only in relations between Member States. That is why, in the return proceedings initiated at the father’s request on 7 July 2017 through the Federal Office of Justice in Berne, which sought the return of the child to Switzerland, the requirements arising from Article 11 of Regulation No 2201/2003, relating to the implementation of procedures under the 1980 Hague Convention, do not apply, since the Swiss Confederation is not bound by Regulation No 2201/2003.\n\n34\n\nConsequently, according to the referring court, following the refusal of the application for return, the Polish court had no reason to proceed in accordance with Article 11(6) and (7) of that regulation and to inform the German courts or central authority of the order on non-return. The referring court adds that the second application for return, which the father lodged with the Federal Office of Justice in Bonn shortly before the lodging of his application for sole custody giving rise to the present proceedings, cannot serve as a basis for continuing to have jurisdiction under Article 10 of that regulation, since the father did not pursue that application for return.\n\n35\n\nIn the third place, the referring court considers that, in so far as Article 10 of Regulation No 2201/2003 is applicable in the present case, the conditions for the application of Article 10(b)(i) of that regulation, which provides that the courts of the Member State in which the child was habitually resident immediately before the wrongful removal or retention will retain jurisdiction, are not, in principle, satisfied. Although the father claims that the child was wrongfully removed to Poland in May 2017, his application for rights of custody was not made until 12 July 2018, with the result that the period of one year laid down in Article 10(b)(i) of that regulation was not observed. That period could, however, be complied with if it began to run on the date on which that child, according to her father, had to attend a kindergarten in Switzerland, namely from November 2017.\n\n36\n\nThe referring court notes, however, that the account of the facts given by the father in that regard in the present proceedings differs from that which he presented during the proceedings under the 1980 Hague Convention. The question therefore arises as to whether the father is time-barred from submitting new facts as to the exact date of the wrongful removal and whether the rules on the burden of proof applicable in proceedings under that convention can be transposed to the present proceedings. The referring court is inclined to take the view that it is not bound by the decision on the application for return made under that convention and that it must assess the contradictions in the father’s account of the facts.\n\n37\n\nFinally, in the fourth place, the referring court observes that, in the event of a refusal to return the child under Article 13 of the 1980 Hague Convention, the rules of Article 11(6) to (8) of Regulation No 2201/2003 encourage proceedings relating to the custody of the child to be brought before the courts of the Member State where the child was habitually resident immediately before the wrongful removal or retention. However, contrary to the father’s submissions, the referring court is of the opinion that the application of the provisions of Article 11 necessarily presupposes the implementation of a procedure under the 1980 Hague Convention between two Member States bound by Regulation No 2201/2003, which is not the case here.\n\n38\n\nIn those circumstances, the Oberlandesgericht Frankfurt am Main (Higher Regional Court, Frankfurt am Main) decided to stay the proceedings and to refer the following questions to the Court of Justice for a preliminary ruling:\n\n‘To what extent is the regulatory mechanism provided for in [Articles 10 and 11] of [Regulation No 2201/2003] limited to proceedings conducted in the context of relations between EU Member States?\n\nMore specifically:\n\n(1)\n\nDoes Article 10 of [Regulation No 2201/2003] apply, with the effect that the jurisdiction of the courts in the former [Member] State of residence is retained, if the child had his or her habitual residence in an EU Member State (Germany) before his or her removal and the return proceedings under the [1980 Hague Convention] were conducted between an EU Member State (Poland) and a third State (Switzerland) and, in those proceedings, the return of the child was refused?\n\nIf Question 1 is answered in the affirmative:\n\n(2)\n\nIn the context of Article 10(b)(i) of [Regulation No 2201/2003], what requirements are to be imposed for the purposes of establishing continuing jurisdiction [of the courts of the Member State of the former habitual residence of the child]?\n\n(3)\n\nDoes Article 11(6) to (8) of [Regulation No 2201/2003] also apply in the case of return proceedings implemented under the [1980 Hague Convention] in the context of relations between a third State and an EU Member State, as a State [to which the child has been removed], in so far as the child had his or her habitual residence in another EU Member State before the removal?’\n\nConsideration of the questions referred\n\nThe first question\n\n39\n\nAccording to settled case-law, in the procedure laid down by Article 267 TFEU providing for cooperation between national courts and the Court of Justice, it is for the latter to provide the national court with an answer which will be of use to it and enable it to decide the case before it. To that end, the Court of Justice should, where necessary, reformulate the questions referred to it (judgment of 30 January 2024, Direktor na Glavna direktsia Natsionalna politsia pri MVR – Sofia, \nC‐118/22\n, \nEU:C:2024:97\n, paragraph \n31\n and the case-law cited).\n\n40\n\nIn the present case, the first question arises from the fact that, according to the referring court, the application of Article 10 of Regulation No 2201/2003 is conditional on proceedings for the return of the child having been commenced under the 1980 Hague Convention, between two Member States, as supplemented by the provisions of Article 11 of that regulation. In so far as the father, before the dispute in the present case, initiated proceedings for the return of the child through the Central Authority of the Swiss Confederation, a third country which, it is common ground, is not bound by Regulation No 2201/2003, the referring court takes the view that neither the provisions of Article 11 nor, consequently, those of Article 10 thereof are applicable in the main proceedings.\n\n41\n\nIn such circumstances, the referring court has doubts as to whether the German courts retain jurisdiction as courts of the Member State of habitual residence of the child immediately before the wrongful removal or retention.\n\n42\n\nIt follows that, by its first question, the referring court seeks, in essence, to ascertain whether Article 10(b)(i) of Regulation No 2201/2003 must be interpreted as meaning that that provision ceases to be applicable on the sole ground that a request was made to the central authority of a third country for proceedings to be commenced for the return of the child under the 1980 Hague Convention and that those proceedings have failed.\n\n43\n\nWithout calling into question the admissibility of that question, the Polish Government submits that Article 10 is not applicable to the dispute in the main proceedings in so far as a Polish court has dismissed the application made by L’s father seeking an order for the return of his child, pursuant to the 1980 Hague Convention, finding that there had been no wrongful removal or retention of that child.\n\n44\n\nIn that regard, it is sufficient to note that, as is confirmed by Article 11(8) of Regulation No 2201/2003, the decision of a court of a Member State refusing to grant an application for return under the 1980 Hague Convention does not preclude a court of another Member State from assuming jurisdiction on the basis of Article 10 of that regulation.\n\n45\n\nIn the light of that clarification, it should be noted that under Article 8(1) of Regulation No 2201/2003, general jurisdiction in matters of parental responsibility is conferred on the courts of the Member State in which the child is habitually resident at the time the court is seised. Because of their geographical proximity, those courts are generally the best placed to assess the measures to be taken in the interests of the child (judgment of 14 July 2022, CC (Transfer of a child’s habitual residence to a third country), \nC‐572/21\n, \nEU:C:2022:562\n, paragraph \n27\n and the case-law cited).\n\n46\n\nHowever, in accordance with Article 8(2) of that regulation, that general jurisdiction is to apply ‘subject to the provisions of Articles 9, 10 and 12’ of that regulation.\n\n47\n\nArticle 10 of Regulation No 2201/2003 provides that the courts of the Member State where the child was habitually resident immediately before the wrongful removal or retention are to retain their jurisdiction until the child has acquired a habitual residence in another Member State.\n\n48\n\nThe transfer of jurisdiction to the courts of that other Member State is subject to the condition, set out in Article 10(a) of that regulation, that a person with custody rights has given his or her consent to that removal or retention, or to the conditions set out in Article 10(b) thereof. In accordance with that paragraph (b), it is required, first, that the child has resided in that Member State for a period of at least one year after the person, institution or other body having rights of custody has had or should have had knowledge of the whereabouts of the child, second, that the child is settled in his or her new environment and, third, that one of the four other conditions set out in paragraphs (i) to (iv) of that provision is met. The condition laid down in paragraph (i) of that provision provides that, within one year after the holder of rights of custody has had or should have had knowledge of the whereabouts of the child, ‘no request for return has been lodged before the competent authorities of the Member State where the child has been removed or is being retained’.\n\n49\n\nFurthermore, it must be borne in mind that, under Article 11(1) of Regulation No 2201/2003, where a person having rights of custody applies to the competent authorities in a Member State to deliver a judgment on the basis of the 1980 Hague Convention in order to obtain the return of a child that has been wrongfully removed or retained in a Member State other than the Member State where the child was habitually resident immediately before the wrongful removal or retention, paragraphs 2 to 8 of Article 11 are to apply.\n\n50\n\nIt is clear from the wording of that Article 11 that that provision applies only where proceedings have been initiated between Member States under the 1980 Hague Convention for the return of a child who has been wrongfully removed or retained.\n\n51\n\nHowever, there is nothing in the wording or in the scheme of Article 10 of that regulation or in the objectives pursued by it to support the argument that the rule of special jurisdiction laid down in Article 10 consisting, in principle, in maintaining the jurisdiction of the courts of the Member State where the child was habitually resident immediately before the wrongful removal or retention becomes inapplicable on the ground that return proceedings have been unsuccessfully initiated under the 1980 Hague Convention between the central or judicial authorities of a third country and a Member State.\n\n52\n\nIn the first place, it must be recalled that the rule on jurisdiction laid down in Article 10 of Regulation No 2201/2003 is based on ‘the wrongful removal or retention of a child’, understood as, in accordance with Article 2(11) of that regulation, a removal or retention in breach of rights of custody acquired by judgment or by operation of law or by an agreement having legal effect under the law of the Member State where the child was habitually resident immediately before the removal or retention, provided that, at the time of removal or retention, the rights of custody were actually exercised, either jointly or alone, or would have been so exercised but for the removal or retention (see, to that effect, judgment of 2 August 2021, A, \nC‐262/21 PPU\n, \nEU:C:2021:640\n, paragraph \n44\n).\n\n53\n\nThat definition of the wrongful removal or retention of a child thus merely refers to an infringement of the rights of custody of one of the holders of parental responsibility under the law of the Member State of the child’s habitual residence immediately before that removal or retention. It does not therefore depend on the holder of rights of custody initiating proceedings, which, as the case may be, would have to take place subsequently, for the return of the child under the 1980 Hague Convention.\n\n54\n\nSuch an interpretation is supported by the objective pursued by Article 10 of Regulation No 2201/2003, which is to avoid giving a procedural advantage to the perpetrator of the wrongful abduction of the child which would result from the fact that the courts of the Member State in which that child was habitually resident immediately before that abduction would automatically lose their jurisdiction on the sole ground that that child is now habitually resident with the perpetrator of that abduction in another Member State (see, to that effect, judgment of 13 July 2023, TT (Wrongful removal of a child), \nC‐87/22\n, \nEU:C:2023:571\n, paragraph \n36\n and the case-law cited).\n\n55\n\nIn the second place, while Article 10(b) refers to the absence of any request for return to the competent authorities of the Member State on whose territory the child has been wrongfully removed or retained, in order to terminate the jurisdiction of the courts of the child’s former habitual residence, that provision does not specify that such a request must have been made under the 1980 Hague Convention, nor does it exclude that it could have been made through a central authority of a third country.\n\n56\n\nBy contrast, the premiss on which the referring court relies would amount to requiring the holder of parental responsibility, whose rights of custody have been infringed within the meaning of Article 2(11) of that regulation, to rely on the provisions of the 1980 Hague Convention in order to request the return of the child concerned.\n\n57\n\nIt must be borne in mind that, first, in accordance with Article 60 of Regulation No 2201/2003, those provisions do not take precedence over the provisions of the regulation in relations between Member States in matters governed by it (see, to that effect, judgment of 13 July 2023, TT (Wrongful removal of a child), \nC‐87/22\n, \nEU:C:2023:571\n, paragraph \n58\n).\n\n58\n\nSecond, the claim that there was an obligation to rely on the provisions of the 1980 Hague Convention in order to apply for the return of a child subject to international abduction has already been rejected by the Court in the judgment of 19 September 2018, C.E. and N.E. (\nC‐325/18 PPU and C‐375/18 PPU\n, \nEU:C:2018:739\n, paragraphs \n49\n and \n51\n). As is apparent from Article 34 of that convention, a return procedure may be based on other rules or other provisions of international agreements, in particular, bilateral provisions. In that regard, the Court has also stated in paragraph 53 of that judgment that the holder of parental responsibility may apply for the recognition and enforcement, in accordance with the provisions of Chapter III of Regulation No 2201/2003, of a decision relating to parental authority and the return of children that has been made by a court having jurisdiction under Chapter II, Section 2, of that regulation, even if that holder of parental responsibility has not submitted an application for return under the 1980 Hague Convention.\n\n59\n\nAccordingly, the mere fact that the parent whose right of custody has been infringed has unsuccessfully initiated proceedings under the 1980 Hague Convention for the return of the child wrongfully removed or retained through the central authority of a third country, and subsequently transmitted to the competent authorities of a Member State, has no bearing on the application to such a situation of the rules on jurisdiction laid down in Article 10 of Regulation No 2201/2003.\n\n60\n\nIn the third place, contrary to what the referring court maintains, the judgment of 24 March 2021, MCP (\nC‐603/20 PPU\n, \nEU:C:2021:231\n), by which the Court held that Article 10 of that regulation does not apply to a situation in which a child has acquired, on the date on which the application relating to parental responsibility was lodged, his or her habitual residence in a third country following abduction to that country, is irrelevant to the foregoing interpretation. It is common ground in the main proceedings that the allegedly wrongful removal took place between two Member States, a situation which does indeed fall within the scope of that provision.\n\n61\n\nFinally, in the fourth place, contrary to what the German Government claims, it cannot be accepted, given that Regulation No 2201/2003 is silent on the matter, that the application of the rule on jurisdiction in matters of parental responsibility, laid down in Article 10 of that regulation, is subject to the application of procedural rules, such as those set out in Article 11(6) and (7) of that regulation, the main purpose of which is to govern the transmission of information on a judgment of non-return, adopted pursuant to Article 13 of the 1980 Hague Convention, which must be communicated to the court with jurisdiction in the Member State where the child was habitually resident immediately before the wrongful removal or retention and to determine the arrangements for service of that information (see, to that effect, judgment of 9 January 2015, RG, \nC‐498/14 PPU\n, \nEU:C:2015:3\n, paragraph \n46\n).\n\n62\n\nIn the light of those considerations, the answer to the first question is that Article 10(b)(i) of Regulation No 2201/2003 must be interpreted as meaning that that provision does not cease to be applicable on the sole ground that a request was made to the central authority of a third country for proceedings to be commenced for the return of a child under the 1980 Hague Convention and that those proceedings have failed.\n\nThe second question\n\n63\n\nBy its second question, the referring court asks, in general terms, what conditions must be satisfied in order to establish that the courts of the Member State in which the child was habitually resident immediately before the wrongful removal or retention continue to have jurisdiction.\n\n64\n\nIt is apparent from the statement of reasons and the factual circumstances set out in the request for a preliminary ruling that that question concerns specifically two points relating, in particular, to the concept of a ‘request for return’, as referred to in Article 10(b)(i) of that regulation. First, the referring court takes the view that the application for return lodged by L’s father on 7 July 2017 does not constitute a ‘request for return’ within the meaning of Article 10(b)(i) on the ground that it sought to obtain the return of the child to a third country, namely the Swiss Confederation. Second, that court considers that the application for custody made by the father on 12 July 2018 may be treated in the same way as a ‘request for return’ within the meaning of Article 10(b)(i). It notes, however, that should the \ndies a quo\n of that period be identical to the one applicable in the context of the request for return lodged on 7 July 2017, as per the father’s claims, that request was made after the one-year period laid down by that provision. In that context, the referring court also asks whether the holder of rights of custody is entitled to adduce new evidence in relation to that which he or she has put forward in the context of those proceedings and, in that regard, what the rules are in relation to the burden of proof.\n\n65\n\nIn the light of those clarifications and of the Court’s case-law referred to in paragraph 39 of the present judgment, it is necessary to reformulate the second question with the result that the referring court asks, in essence, whether Article 10(b)(i) of Regulation No 2201/2003 must be interpreted as meaning that the concept of a ‘request for return’, within the meaning of that provision, covers an application request for the return of the child to a State other than the Member State in which the child was habitually resident immediately before the wrongful removal or retention or an application for custody of that child brought before the courts of that Member State. If the answer is in the affirmative, the referring court asks, first, whether, in order to establish that the holder of rights of custody has lodged a request for return within the period laid down by that provision, that holder is entitled to adduce new evidence in relation to the evidence which he or she submitted in the course of the proceedings conducted under the 1980 Hague Convention and, second, whether the rules on the burden of proof are identical to those applicable in those proceedings.\n\n66\n\nAs regards, in the first place, the question whether, as argued in particular by the Commission, a request for return of the child to a State, including to a third country, distinct from the Member State where that child was habitually resident immediately before the wrongful removal or retention falls within Article 10(b)(i) of Regulation No 2201/2003, it should be noted, first of all, that that regulation does not specify what is to be understood by ‘request for return’.\n\n67\n\nAccording to settled case-law, in interpreting a provision of EU law, it is appropriate to consider not only its wording but also its context and the objectives pursued by the legislation of which it forms part (see, to that effect, judgment of 13 July 2023, TT (Wrongful removal of a child), \nC‐87/22\n, \nEU:C:2023:571\n, paragraph \n39\n and the case-law cited).\n\n68\n\nIn that regard, first, there is nothing in the wording of Article 10 of Regulation No 2201/2003 to suggest that the expression ‘request for return’ refers to an action other than that by which a person seeks the return of a child to the Member State on whose territory he or she was habitually resident immediately before the wrongful removal or retention.\n\n69\n\nNext, as regards the context of Article 10 of that regulation, it should be borne in mind that that article lays down a special rule on jurisdiction in relation to the general rule laid down in Article 8(1) of that regulation. Article 10 thus sets out the circumstances in which jurisdiction in favour of the courts of the Member State where the child was habitually resident immediately before the wrongful removal or retention continues or, on the contrary, is transferred to the courts of the Member State where the child has acquired habitual residence following the wrongful removal or retention.\n\n70\n\nIt is therefore logical and consistent with the scheme of the rules of jurisdiction in matters of parental responsibility laid down in Regulation No 2201/2003 that, first, the ‘request for return’, referred to in Article 10 of that regulation, must be made to the competent authorities of the Member State in whose territory the child has been wrongfully removed and is physically present and, second, that same request seeks to obtain the return of that child to the Member State in whose territory the child was habitually resident immediately before the wrongful removal, and whose courts are, as the Court has already held, on account of their geographical proximity, generally best placed to assess the measures to be taken in the interests of that child (see, to that effect, judgment of 13 July 2023, TT (Wrongful removal of a child), \nC‐87/22\n, \nEU:C:2023:571\n, paragraph \n33\n and the case-law cited). A request for the child to be taken to another State, which is, moreover, a third country, in whose territory the child was not habitually resident before the wrongful removal, goes against that logic.\n\n71\n\nLastly, that interpretation is supported by the objective of Regulation No 2201/2003. That regulation seeks to deter child abductions between States and, in cases of abduction, obtain the child’s prompt return to the State of his or her habitual residence (judgment of 19 September 2018, C.E. and N.E., \nC‐325/18 PPU and C‐375/18 PPU\n, \nEU:C:2018:739\n, paragraph \n47\n).\n\n72\n\nFurthermore, the Court has already held, with regard to the interpretation of Article 11 of Regulation No 2201/2003, that one of the objectives of that provision is the restoration of the \nstatus quo ante\n, that is to say, the situation that existed prior to the wrongful removal or retention of the child (see, to that effect, judgment of 16 February 2023, Rzecznik Praw Dziecka and Others (Suspension of the return decision), \nC‐638/22 PPU\n, \nEU:C:2023:103\n, paragraph \n69\n and the case-law cited).\n\n73\n\nEven if, as has been held in paragraphs 51 to 62 of the present judgment, the jurisdiction established by Article 10 of Regulation No 2201/2003 is not conditional on the initiation of return proceedings under the 1980 Hague Convention, as supplemented, as between the Member States, by the provisions of Article 11 of that regulation, the fact remains that the restoration of \nthe status quo\n\nante\n necessarily constitutes a common objective of the requests for return referred to in Articles 10 and 11 of that regulation.\n\n74\n\nAccordingly, all those objectives would be undermined if a ‘request for return’ were to be understood as an application to transfer the child to a State in whose territory that child was not habitually resident immediately before the wrongful removal or retention (see, to that effect, judgment of 8 June 2017, OL, \nC‐111/17 PPU\n, \nEU:C:2017:436\n, paragraph \n38\n).\n\n75\n\nThat interpretation is supported by the 1980 Hague Convention. Although it is true, as the Commission submits, that the first paragraph of Article 8 of that convention authorises the holder of rights of custody to make an application for return through the Central Authority of any Contracting Party, the preamble to that convention nevertheless states that the convention is intended to protect children internationally against the harmful effects of wrongful removal or retention and to establish procedures to ensure the prompt return of the child to the State of his or her habitual residence (see, to that effect, judgment of 16 February 2023, Rzecznik Praw Dziecka and Others (Suspension of the return decision), \nC‐638/22 PPU\n, \nEU:C:2023:103\n, paragraph \n64\n).\n\n76\n\nIt thus follows from the literal, contextual and teleological interpretation of Article 10(b)(i) of Regulation No 2201/2003 that the concept of a ‘request for return’, within the meaning of that provision, refers to an application by which a person requests that a child return to the Member State in whose territory he or she was habitually resident immediately before the wrongful removal or retention.\n\n77\n\nConversely, an application for the child to join one of his or her parents in a third country where that child was not habitually resident immediately before the child was wrongfully removed does not constitute a ‘request for return’ within the meaning of Article 10(b)(i).\n\n78\n\nIn the second place, an application for custody before the courts of the Member State where the child was habitually resident immediately before the wrongful removal or retention cannot be regarded as equivalent to a request for return within the meaning of Article 10(b)(i) of Regulation No 2201/2003.\n\n79\n\nAs the Advocate General observed, in essence, in point 61 of his Opinion, it is apparent from Article 10(b) of that regulation that a request for the return of a child and an application for custody of a child are not interchangeable, since those two applications have different functions. First, unlike an application for custody of a child, which requires an in-depth examination of the substance of the dispute in matters of parental responsibility, a request for return is, by its nature, the subject of an expedited procedure, since it seeks to ensure, as stated in recital 17 of Regulation No 2201/2003, the return of the child without delay (see, to that effect, judgment of 16 February 2023, Rzecznik Praw Dziecka and Others (Suspension of the return decision), \nC‐638/22 PPU\n, \nEU:C:2023:103\n, paragraphs \n68\n and \n70\n). Second, the Court has already held that a decision on the return or the non-return of the child does not settle the issue of rights of custody with respect to that child, it being understood that a failure to succeed in a return procedure is without prejudice to the possibility, for the parent whose rights of custody have been infringed, of asserting his or her rights on the substance of parental responsibility by means of proceedings before the courts with the relevant jurisdiction under the provisions of Regulation No 2201/2003 (see, to that effect, judgment of 8 June 2017, OL, \nC‐111/17 PPU\n, \nEU:C:2017:436\n, paragraph \n65\n and the case-law cited).\n\n80\n\nSince neither an application for the return of a child to a State in whose territory that child was not habitually resident immediately before the wrongful removal or retention nor an application for custody made in respect of that child can be classified as ‘request(s) for return’ within the meaning of Article 10(b)(i) of Regulation No 2201/2003, there is no need to examine the questions referred to in the last sentence of paragraph 65 above.\n\n81\n\nIn the light of the foregoing considerations, the answer to the second question is that Article 10(b)(i) of Regulation No 2201/2003 must be interpreted as meaning that the concept of ‘request for return’ within the meaning of that provision does not cover either an application for the return of the child to a State other than the Member State in which that child was habitually resident immediately before the wrongful removal or retention, or an application for custody of that child brought before the courts of that Member State.\n\nThe third question\n\n82\n\nBy its third question, the referring court asks, in essence, whether Article 11(6) to (8) of Regulation No 2201/2003 must be interpreted as meaning that that provision applies where proceedings for the return of a child under the 1980 Hague Convention have commenced between a third country and a Member State on whose territory that child is present following a wrongful removal or retention, in so far as the child was habitually resident in another Member State before being removed.\n\n83\n\nAs has been stated in paragraph 50 of the present judgment, it follows from the wording of Article 11 of that regulation that it applies only in conjunction with the provisions of the 1980 Hague Convention in relations between the Member States.\n\n84\n\nIt follows that, as the German and Polish Governments and the Commission have rightly submitted, the obligations to inform and notify laid down in Article 11(6) and (7) of that regulation and the enforceability of the judgment referred to in Article 11(8) of that regulation do not apply in the context of proceedings for the return of the child which have commenced between a central authority of a third country and the authorities of the Member State in which that child is present following a wrongful removal or retention.\n\n85\n\nIn the light of those considerations, the answer to the third question is that Article 11(6) to (8) of Regulation No 2201/2003 must be interpreted as not applying when proceedings for the return of a child under the 1980 Hague Convention have commenced between a third country and a Member State on the territory of which that child is present following a wrongful removal or retention.\n\nCosts\n\n86\n\nSince these proceedings are, for the parties to the main proceedings, a step in the action pending before the referring court, the decision on costs is a matter for that court. Costs incurred in submitting observations to the Court, other than the costs of those parties, are not recoverable.\n\n \n\nOn those grounds, the Court (Fourth Chamber) hereby rules:\n\n \n\n1.\n\nArticle 10(b)(i) of Council Regulation (EC) No 2201/2003 of 27 November 2003 concerning jurisdiction and the recognition and enforcement of judgments in matrimonial matters and the matters of parental responsibility, repealing Regulation (EC) No 1347/2000,\n\nmust be interpreted as meaning that that provision does not cease to be applicable on the sole ground that a request was made to the central authority of a third country for proceedings to be commenced for the return of a child under the Convention on the Civil Aspects of International Child Abduction, concluded in The Hague on 25 October 1980, and that those proceedings have failed.\n\n \n\n2.\n\nArticle 10(b)(i) of Regulation No 2201/2003\n\nmust be interpreted as meaning that the concept of ‘request for return’ within the meaning of that provision does not cover either an application for the return of the child to a State other than the Member State in which that child was habitually resident immediately before the wrongful removal or retention, or an application for custody of that child brought before the courts of that Member State.\n\n \n\n3.\n\nArticle 11(6) to (8) of Regulation No 2201/2003\n\nmust be interpreted as not applying when proceedings for the return of a child under the Convention on the Civil Aspects of International Child Abduction, concluded in The Hague on 25 October 1980 have commenced between a third country and a Member State on the territory of which that child is present following a wrongful removal or retention.\n\n \n\n[Signatures]\n\n(\n\n*1\n\n)\tLanguage of the case: German.\n\n(\n\n1\n\n)\tThe name of the present case is a fictitious name. It does not correspond to the real name of any of the parties to the proceedings.
<<<END OF RAW TEXT>>>

#### Filled-out final JSON schema of example 2
{
  "case_metadata": {
    "celex": "62023CJ0035",
    "document_link": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=cellar:bb2daadb-2ee7-11ef-a61b-01aa75ed71a1",
    "ecli": "ECLI:EU:C:2024:532",
    "case_number": "C-35/23",
    "date_of_judgment": "2024-06-20",
    "document_type": "Judgment",
    "court": "Court of Justice",
    "case_name": "Père v Mère"
  },
  "case_summary": {
    "subject_matter": "Interpretation of jurisdiction rules under Regulation No 2201/2003 regarding the wrongful removal of a child and the definition of a 'request for return' involving a third country.",
    "keywords": [
      {
        "term": "Parental responsibility",
        "description": "Legal rights and duties concerning a child's person or property, including custody and access rights."
      },
      {
        "term": "Child abduction",
        "description": "The wrongful removal or retention of a child in breach of custody rights established by law or agreement."
      },
      {
        "term": "Habitual residence",
        "description": "The primary jurisdictional criterion for determining which court is best placed to decide on matters of parental responsibility."
      },
      {
        "term": "1980 Hague Convention",
        "description": "International treaty establishing procedures for the prompt return of children wrongfully removed to another contracting state."
      }
    ],
    "summary_text": "The case originated from a dispute between a German father and a Polish mother regarding their child, L. After living in Germany, the mother moved the child to Poland. The father initially sought the child's return to Switzerland (his place of residence) under the 1980 Hague Convention, but Polish courts refused. Subsequently, the father applied for sole custody in Germany, claiming German courts retained jurisdiction under Regulation No 2201/2003. The German court referred questions to the CJEU to clarify if jurisdictional protections for the state of former habitual residence apply when return proceedings involve a third country. The Court held that Article 10(b)(i) of the Regulation remains applicable even if a return request was unsuccessfully made to a third country's central authority. However, it clarified that a 'request for return' under this provision specifically means seeking the child's return to the Member State of prior habitual residence, not to a third country. Furthermore, an application for custody does not qualify as a request for return. Finally, the Court ruled that the procedural mechanisms in Article 11(6) to (8) of the Regulation, which facilitate information exchange and enforcement between Member States, do not apply to return proceedings conducted between a Member State and a third country like Switzerland."
  },
  "case_segments": {
    "document_start": [
      {
        "start_anchor": "62023CJ0035\n\n \nJUDGMENT OF THE COURT (Fourth Chamber)\n\n20 June 2024 (\n\n*1\n\n)\n\n(Reference for a preliminary ruling – Judicial cooperation in civil matters – Parental responsibility – Regulation (EC) No 2201/2003 – Articles 10"
      }
    ],
    "procedural_introduction": [
      {
        "start_anchor": "Judgment\n\n1\n\nThis request for a preliminary ruling concerns the interpretation of Articles 10 and 11 of Council Regulation (EC) No 2201/2003 of 27 November 2003 concerning jurisdiction and the recognition"
      }
    ],
    "legal_context": [
      {
        "start_anchor": "Legal context\n\nThe 1980 Hague Convention\n\n3\n\nAs set out in the preamble to the Convention on the Civil Aspects of International Child Abduction, concluded in The Hague on 25 October 1980"
      }
    ],
    "facts_and_background": [
      {
        "start_anchor": "The dispute in the main proceedings and the questions referred for a preliminary ruling\n\n16\n\nL was born in Switzerland in November 2014 and has dual German and Polish nationality."
      }
    ],
    "reasoning": [
      {
        "start_anchor": "Consideration of the questions referred\n\nThe first question\n\n39\n\nAccording to settled case-law, in the procedure laid down by Article 267 TFEU providing for cooperation between national courts and the Court"
      }
    ],
    "decision_on_costs": [
      {
        "start_anchor": "Costs\n\n86\n\nSince these proceedings are, for the parties to the main proceedings, a step in the action pending before the referring court, the decision on costs is a matter for"
      }
    ],
    "operative_part": [
      {
        "start_anchor": "On those grounds, the Court (Fourth Chamber) hereby rules:\n\n \n\n1.\n\nArticle 10(b)(i) of Council Regulation (EC) No 2201/2003 of 27 November 2003 concerning jurisdiction and the recognition and enforcement of"
      }
    ],
    "document_end": [
      {
        "start_anchor": "[Signatures]\n\n(\n\n*1\n\n)\tLanguage of the case: German.\n\n(\n\n1\n\n)\tThe name of the present case is a fictitious name. It does not correspond to the real name of any of the parties"
      }
    ]
  }
}

---
""".strip()


@dataclass
class GenerateConfig:
    model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.1
    thinking_budget: int = 8192
    max_attempts: int = 3
    base_delay: float = 60.0
    max_delay: float = 120.0
    jitter: float = 0.10


def _safe_string(x: Any) -> str:
    try:
        if x is None or pd.isna(x):
            return "unknown"
    except Exception:
        pass
    if isinstance(x, pd.Timestamp):
        return x.date().isoformat()
    return str(x)


def _raw_text(x: Any) -> str:
    try:
        if x is None or pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def escape_control_chars_in_json_strings(s: str) -> str:
    if not s:
        return s
    control_map = {
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
        "\b": "\\b",
        "\f": "\\f",
    }
    out: List[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                out.append(ch)
                esc = False
            else:
                if ch == "\\":
                    out.append(ch)
                    esc = True
                elif ch == '"':
                    out.append(ch)
                    in_str = False
                elif ch in control_map:
                    out.append(control_map[ch])
                elif ord(ch) < 0x20:
                    out.append("\\u%04x" % ord(ch))
                else:
                    out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_str = True
    return "".join(out)


def validate_json_text(text: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    if not text:
        return False, None, "Empty output"
    try:
        return True, json.loads(text), None
    except json.JSONDecodeError as e:
        fixed = escape_control_chars_in_json_strings(text)
        if fixed != text:
            try:
                return True, json.loads(fixed), None
            except Exception as e2:
                return False, None, f"{e} (repair failed: {e2})"
        return False, None, str(e)
    except Exception as e:
        return False, None, str(e)


def save_failure(celex: str, raw: str, err: str, fail_dir: Path) -> Tuple[Path, Path, Path]:
    ensure_dir(fail_dir)
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in celex)
    raw_path = fail_dir / f"{safe}.raw.txt"
    err_path = fail_dir / f"{safe}.error.txt"
    repr_path = fail_dir / f"{safe}.repr.txt"
    raw_path.write_text(raw or "", encoding="utf-8")
    err_path.write_text(err or "", encoding="utf-8")
    repr_path.write_text(repr(raw or ""), encoding="utf-8")
    return raw_path, err_path, repr_path


class GeminiExtractor:
    def __init__(self, config: GenerateConfig | None = None):
        self.config = config or GenerateConfig()
        if genai is None or types is None:
            raise ImportError(
                "google-genai is required for generation. Install it and configure credentials before running this script."
            )
        self.client = genai.Client()

    def run(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.config.model,
            config=types.GenerateContentConfig(
                temperature=self.config.temperature,
                thinking_config=types.ThinkingConfig(thinking_budget=self.config.thinking_budget),
                response_mime_type="application/json",
            ),
            contents=prompt,
        )
        return response.text

    def run_with_retries(self, prompt: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], Optional[str]]:
        last_err: Optional[str] = None
        last_text: Optional[str] = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                last_text = self.run(prompt)
                ok, parsed, err = validate_json_text(last_text)
                if ok:
                    return True, last_text, parsed, None
                last_err = f"Attempt {attempt}/{self.config.max_attempts}: {err}"
            except Exception as e:
                last_err = f"Attempt {attempt}/{self.config.max_attempts}: {type(e).__name__}: {e}"
            if attempt < self.config.max_attempts:
                delay = min(self.config.max_delay, self.config.base_delay * (2 ** (attempt - 1)))
                delay *= 1 + random.uniform(-self.config.jitter, self.config.jitter)
                print(f"Retrying after {delay:.2f} seconds ({last_err})")
                time.sleep(max(0.0, delay))
        return False, last_text, None, last_err


def build_prompt(row: pd.Series) -> str:
    metadata = {
        "celex": _safe_string(row.get("celex")),
        "document_link": _safe_string(row.get("document_link")),
        "ecli": _safe_string(row.get("ecli")),
        "title": _safe_string(row.get("title")),
        "date_of_judgment": _safe_string(row.get("date_of_judgment", row.get("date"))),
        "document_type": _safe_string(row.get("document_type", row.get("resource_type"))),
        "court": _safe_string(row.get("court")),
    }
    raw = _raw_text(row.get("raw_content"))
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
    return (
        f"{BASE_PROMPT}\n\n"
        f"Case metadata block:\n{metadata_json}\n\n"
        f"Raw text of the judgment:\n<<<BEGINNING OF RAW TEXT>>>\n{raw}\n<<<END OF RAW TEXT>>>"
    )


def iter_records(df: pd.DataFrame, celex_col: str = "celex") -> Iterable[Tuple[int, pd.Series, str]]:
    for idx, row in df.iterrows():
        celex = str(row.get(celex_col, "")).strip()
        if not celex:
            celex = str(idx)
        yield idx, row, celex


def generate_json_from_dataframe(
    df: pd.DataFrame,
    output_dir: Path,
    fail_dir: Optional[Path] = None,
    checkpoint_pkl: Optional[Path] = None,
    final_pkl: Optional[Path] = None,
    celex_col: str = "celex",
    overwrite: bool = False,
    model: str = "gemini-2.5-flash-lite",
) -> pd.DataFrame:
    load_dotenv()
    ensure_dir(output_dir)
    fail_dir = fail_dir or output_dir / "_fails"
    ensure_dir(fail_dir)

    extractor = GeminiExtractor(GenerateConfig(model=model))
    result_df = df.copy()
    json_col = "generated_json"
    if json_col not in result_df.columns:
        result_df[json_col] = None

    for idx, row, celex in iter_records(result_df, celex_col=celex_col):
        out_path = output_dir / f"{celex}.json"
        if out_path.exists() and not overwrite:
            try:
                result_df.at[idx, json_col] = load_json(out_path)
                print(f"[SKIP] {celex}: JSON already exists")
                continue
            except Exception:
                pass

        prompt = build_prompt(row)
        ok, raw_text, parsed, err = extractor.run_with_retries(prompt)
        if not ok or parsed is None:
            save_failure(celex, raw_text or "", err or "Unknown error", fail_dir)
            print(f"[FAIL] {celex}: {err}")
            continue

        save_json(out_path, parsed)
        result_df.at[idx, json_col] = parsed
        print(f"[OK] {celex} -> {out_path}")

        if checkpoint_pkl is not None:
            ensure_dir(checkpoint_pkl.parent)
            result_df.to_pickle(checkpoint_pkl)

    if final_pkl is not None:
        ensure_dir(final_pkl.parent)
        result_df.to_pickle(final_pkl)

    return result_df


def _get_case_segments_obj(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if isinstance(data.get("case_segments"), dict):
        return data["case_segments"]
    if all(k in data for k in SEGMENT_ORDER):
        return data
    return None


def _detect_segment_format(seg_val: Any) -> str:
    if isinstance(seg_val, dict) and "start_index" in seg_val and "end_index" in seg_val:
        return "index_dict"
    if isinstance(seg_val, list) and len(seg_val) == 1 and isinstance(seg_val[0], dict):
        d = seg_val[0]
        if "start_index" in d and "end_index" in d:
            return "index_array"
        if "start_anchor" in d:
            return "anchor_array"
    return "unknown"


def _anchor_to_whitespace_flexible_pattern(snippet: str) -> re.Pattern[str]:
    parts = [p for p in re.split(r"\s+", snippet.strip()) if p]
    if not parts:
        return re.compile(r"", flags=re.DOTALL)
    pat = r"\s+".join(re.escape(p) for p in parts)
    return re.compile(pat, flags=re.DOTALL)


def _snippet50_from_start_anchor(seg_val: Any) -> Tuple[Optional[str], bool]:
    if not (isinstance(seg_val, list) and len(seg_val) == 1 and isinstance(seg_val[0], dict)):
        return None, False
    d = seg_val[0]
    if "start_anchor" not in d:
        return None, False
    start_anchor = d.get("start_anchor")
    if start_anchor is None:
        return None, True
    return str(start_anchor)[:50], False


def _bounds_from_indices(raw: str, segments: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
    bounds: Dict[str, Tuple[int, int]] = {}
    n = len(raw)
    for name in SEGMENT_ORDER:
        seg_val = segments.get(name, {})
        fmt = _detect_segment_format(seg_val)
        if fmt == "index_dict":
            si = int(seg_val.get("start_index", 0))
            ei = int(seg_val.get("end_index", 0))
        elif fmt == "index_array":
            d = seg_val[0]
            si = int(d.get("start_index", 0))
            ei = int(d.get("end_index", 0))
        else:
            si, ei = 0, 0
        si = max(0, min(si, n))
        ei = max(0, min(ei, n))
        if ei < si:
            ei = si
        bounds[name] = (si, ei)
    return bounds


def _next_non_none_start(starts: Dict[str, Optional[int]], order: List[str], i: int, default: int) -> int:
    for j in range(i + 1, len(order)):
        v = starts.get(order[j])
        if v is not None:
            return v
    return default


def _bounds_from_anchor_regex_50(
    raw: str, segments: Dict[str, Any]
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Optional[int]], Dict[str, str]]:
    n = len(raw)
    cursor = 0
    starts: Dict[str, Optional[int]] = {}
    status: Dict[str, str] = {}

    for name in SEGMENT_ORDER:
        if name == "document_start":
            starts[name] = 0
            status[name] = "FORCED"
            cursor = 0
            continue

        seg_val = segments.get(name)
        fmt = _detect_segment_format(seg_val)
        if fmt != "anchor_array":
            starts[name] = None
            status[name] = "MISSING/WRONG FORMAT"
            continue

        snippet50, is_explicit_null = _snippet50_from_start_anchor(seg_val)
        if is_explicit_null:
            starts[name] = None
            status[name] = "NULL ANCHOR"
            continue
        if snippet50 is None or not snippet50.strip():
            starts[name] = None
            status[name] = "MISSING/WRONG FORMAT"
            continue

        pat = _anchor_to_whitespace_flexible_pattern(snippet50)
        m = pat.search(raw, pos=cursor)
        if not m:
            m = re.compile(re.escape(snippet50), flags=re.DOTALL).search(raw, pos=cursor)

        if not m:
            starts[name] = None
            status[name] = "NO MATCH"
        else:
            starts[name] = m.start()
            status[name] = "FOUND"
            cursor = m.start()

    bounds: Dict[str, Tuple[int, int]] = {}
    for i, name in enumerate(SEGMENT_ORDER):
        si = starts.get(name)
        if name == "document_start":
            si = 0
        if si is None:
            bounds[name] = (0, 0)
            continue
        ei = _next_non_none_start(starts, SEGMENT_ORDER, i, n)
        bounds[name] = (si, max(si, ei))

    if starts.get("document_end") is not None:
        de_si, _ = bounds["document_end"]
        bounds["document_end"] = (de_si, n)

    return bounds, starts, status


def restore_full_text_from_df(
    df: pd.DataFrame,
    in_dir: Path,
    out_dir: Path,
    celex_col: str = "celex",
    raw_col: str = "raw_content",
) -> None:
    ensure_dir(out_dir)
    lookup: Dict[str, str] = {}
    for c, r in zip(df[celex_col].astype(str), df[raw_col]):
        if c not in lookup:
            lookup[c] = "" if pd.isna(r) else str(r)

    json_files = sorted(in_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {in_dir}")

    for jp in json_files:
        celex = jp.stem
        raw = lookup.get(str(celex))
        if raw is None:
            print(f"[SKIP] {jp.name}: celex '{celex}' not found in dataframe")
            continue

        raw = str(raw)
        n = len(raw)
        data = load_json(jp)
        segments = _get_case_segments_obj(data)
        if segments is None:
            print(f"[SKIP] {jp.name}: no case_segments found")
            continue

        fmt = _detect_segment_format(segments.get("document_start"))
        starts: Dict[str, Optional[int]] = {k: None for k in SEGMENT_ORDER}
        status: Dict[str, str] = {k: "UNKNOWN" for k in SEGMENT_ORDER}

        if fmt in ("index_dict", "index_array"):
            bounds = _bounds_from_indices(raw, segments)
            for name in SEGMENT_ORDER:
                starts[name] = bounds[name][0]
                si, ei = bounds[name]
                status[name] = "FOUND" if ei > si else "EMPTY SLICE"
        elif fmt == "anchor_array":
            bounds, starts, status = _bounds_from_anchor_regex_50(raw, segments)
        else:
            print(f"[SKIP] {jp.name}: unrecognized segment format for document_start")
            continue

        print(f"\n{jp.name} (mode={fmt})")
        for name in SEGMENT_ORDER:
            st = status.get(name, "UNKNOWN")
            if st == "FOUND":
                si, ei = bounds[name]
                print(f"  - {name}: FOUND at start_index={si} end_index={ei}")
            elif st == "FORCED":
                print(f"  - {name}: FORCED (start_index=0)")
            else:
                print(f"  - {name}: {st}")

        restored: Dict[str, Any] = {}
        for name in SEGMENT_ORDER:
            si, ei = bounds[name]
            if fmt == "anchor_array":
                if starts.get(name) is None and name != "document_start":
                    restored[name] = {"start_index": None, "end_index": None, "text": ""}
                    continue
                if name == "document_start":
                    si = 0
                    ei = bounds[name][1]
                if name == "document_end" and starts.get("document_end") is not None:
                    ei = n
            else:
                si = max(0, min(int(si), n))
                ei = max(0, min(int(ei), n))
                if ei < si:
                    ei = si
            restored[name] = {
                "start_index": int(si),
                "end_index": int(ei),
                "text": raw[int(si):int(ei)],
            }

        data["case_segments"] = restored
        out_file = out_dir / jp.name
        save_json(out_file, data)
        print(f"[OK] wrote -> {out_file}")

    print("\nDone.")


def load_pickle(path: Path) -> pd.DataFrame:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame in {path}, got {type(obj).__name__}")
    return obj


def cmd_generate(args: SimpleNamespace) -> None:
    df = load_pickle(Path(args.input))
    generate_json_from_dataframe(
        df=df,
        output_dir=Path(args.output_dir),
        fail_dir=Path(args.fail_dir) if args.fail_dir else None,
        checkpoint_pkl=Path(args.checkpoint_pkl) if args.checkpoint_pkl else None,
        final_pkl=Path(args.final_pkl) if args.final_pkl else None,
        celex_col=args.celex_col,
        overwrite=args.overwrite,
        model=args.model,
    )


def cmd_restore(args: SimpleNamespace) -> None:
    df = load_pickle(Path(args.input))
    restore_full_text_from_df(
        df=df,
        in_dir=Path(args.in_dir),
        out_dir=Path(args.out_dir),
        celex_col=args.celex_col,
        raw_col=args.raw_col,
    )


def cmd_pipeline(args: SimpleNamespace) -> None:
    df = load_pickle(Path(args.input))
    output_dir = Path(args.output_dir)
    restored_dir = Path(args.restored_dir)

    result_df = generate_json_from_dataframe(
        df=df,
        output_dir=output_dir,
        fail_dir=Path(args.fail_dir) if args.fail_dir else None,
        checkpoint_pkl=Path(args.checkpoint_pkl) if args.checkpoint_pkl else None,
        final_pkl=Path(args.final_pkl) if args.final_pkl else None,
        celex_col=args.celex_col,
        overwrite=args.overwrite,
        model=args.model,
    )
    restore_full_text_from_df(
        df=result_df,
        in_dir=output_dir,
        out_dir=restored_dir,
        celex_col=args.celex_col,
        raw_col=args.raw_col,
    )




# =============================================================================
# IDE configuration
# =============================================================================

PROJECT_ROOT_OVERRIDE: Path | None = None
RUN_MODE = "pipeline"  # generate, restore, or pipeline

INPUT_PICKLE = Path("data/raw/eurlex_raw.pkl")
GENERATED_JSON_DIR = Path("data/intermediate/generated_json")
RESTORED_JSON_DIR = Path("data/intermediate/restored_json")
FAILURE_DIR = Path("data/intermediate/generation_failures")
CHECKPOINT_PICKLE = Path("data/intermediate/generation_checkpoint.pkl")
FINAL_PICKLE = Path("data/intermediate/generation_complete.pkl")

CELEX_COLUMN = "celex"
RAW_TEXT_COLUMN = "raw_content"
GEMINI_MODEL = "gemini-2.5-flash-lite"
OVERWRITE_EXISTING_JSON = False


def build_ide_arguments(project_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        input=str(resolve_project_path(project_root, INPUT_PICKLE)),
        output_dir=str(resolve_project_path(project_root, GENERATED_JSON_DIR)),
        restored_dir=str(resolve_project_path(project_root, RESTORED_JSON_DIR)),
        in_dir=str(resolve_project_path(project_root, GENERATED_JSON_DIR)),
        out_dir=str(resolve_project_path(project_root, RESTORED_JSON_DIR)),
        fail_dir=str(resolve_project_path(project_root, FAILURE_DIR)),
        checkpoint_pkl=str(resolve_project_path(project_root, CHECKPOINT_PICKLE)),
        final_pkl=str(resolve_project_path(project_root, FINAL_PICKLE)),
        celex_col=CELEX_COLUMN,
        raw_col=RAW_TEXT_COLUMN,
        model=GEMINI_MODEL,
        overwrite=OVERWRITE_EXISTING_JSON,
    )


def main() -> int:
    set_reproducible_seed(123)
    project_root = find_project_root(
        PROJECT_ROOT_OVERRIDE,
        script_file=globals().get("__file__"),
    )
    args = build_ide_arguments(project_root)
    mode = RUN_MODE.strip().lower()

    try:
        if mode in {"generate", "pipeline"}:
            require_any_environment_variable(("GEMINI_API_KEY", "GOOGLE_API_KEY"))
        if mode == "generate":
            cmd_generate(args)
        elif mode == "restore":
            cmd_restore(args)
        elif mode == "pipeline":
            cmd_pipeline(args)
        else:
            raise ValueError("RUN_MODE must be generate, restore, or pipeline.")
        return 0
    except (EnvironmentError, FileNotFoundError, ValueError, TypeError, OSError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    main()
