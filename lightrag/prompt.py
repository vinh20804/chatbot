from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS[
    "entity_extraction_system_prompt"
] = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Precision Over Recall:** Prefer extracting fewer, high-value entities over capturing every noun phrase. Skip generic, low-information, or presentation-only phrases even if they appear repeatedly.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
        *   `entity_type`: Categorize the entity using one of the following types: `{entity_types}`. If none of the provided entity types apply, **SKIP the entity entirely** (do NOT classify as "Other" or any new type). Maintaining a clean typed graph is more important than capturing every entity.
        *   `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based EXCLUSIVELY on the information present in the input text. If the entity is not explicitly mentioned in the text, SKIP it entirely. Do NOT infer attributes from general knowledge. Do NOT add background context not present in the text.
    *   **Skip These Cases Entirely:** Do NOT extract section titles, slide headings, agenda items, generic academic activity phrases, discourse markers, or broad unlabeled abstractions unless the text explicitly defines them as central technical concepts. This includes phrases like `Nội dung`, `Mục tiêu`, `Đánh giá`, `Phân tích`, `Trao đổi`, `Trình bày`, `Kết luận`, `Giới thiệu`, and similar presentation scaffolding.
    *   **Entity Significance Test:** Extract an entity only if at least one of the following is true:
        *   it is explicitly defined or described in the text,
        *   it participates in a specific technical relationship,
        *   it is a named method, model, framework, algorithm, tool, metric, person, or organization,
        *   it is a domain concept whose identity is more specific than a generic task or document section.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities. If a relationship is not explicitly stated in the text, SKIP it entirely. Do NOT infer relationships from general/background knowledge.
    *   **Conservative N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into binary pairs ONLY when each resulting pair is explicitly meaningful on its own. Do NOT create all pairwise combinations mechanically.
        *   **Example:** For "Alice, Bob, and Carol collaborated on Project X," prefer `Alice -> Project X`, `Bob -> Project X`, `Carol -> Project X` if those are the direct stated relations. Do NOT add `Alice -> Bob` unless the collaboration between them is explicitly stated or clearly central.
    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `relationship_keywords`: One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship. Multiple keywords within this field must be separated by a comma `,`. **DO NOT use `{tuple_delimiter}` for separating multiple keywords within this field.**
        *   `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities, based EXCLUSIVELY on the information present in the input text. Do NOT infer rationale from general knowledge. Do NOT add background context not present in the text.
    *   **Skip These Relationships Entirely:** Do NOT create a relationship only because two entities appear in the same sentence, bullet list, heading, or slide. Do NOT create relationships involving generic academic activity phrases, presentation structure, or document organization unless the relationship is itself the substantive topic.
    *   **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.
    *   If either endpoint is too generic, presentation-like, or weakly defined, SKIP the relationship entirely.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

7.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

---Examples---
{examples}
"""

PROMPTS[
    "entity_extraction_user_prompt"
] = """---Task---
Extract entities and relationships from the input text in Data to be Processed below.

---Instructions---
1.  **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.
2.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
3.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant entities and relationships have been extracted and presented.
4.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

---Data to be Processed---
<Entity_types>
[{entity_types}]

<Input Text>
```
{input_text}
```

<Output>
"""

PROMPTS[
    "entity_continue_extraction_user_prompt"
] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Strict Adherence to System Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system instructions.
2.  **Focus on Corrections/Additions:**
    *   **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
    *   If an entity or relationship was **missed** in the last task, extract and output it now according to the system format.
    *   If an entity or relationship was **truncated, had missing fields, or was otherwise incorrectly formatted** in the last task, re-output the *corrected and complete* version in the specified format.
    *   Maintain the same conservative filtering rules from the system prompt. Do NOT add generic headings, presentation scaffolding, or weakly supported relationships during this continuation step.
3.  **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
4.  **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
5.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
6.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant missing or corrected entities and relationships have been extracted and presented.
7.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

<Output>
"""

PROMPTS["entity_extraction_examples"] = [
    """<Entity_types>
["concept","algorithm","data_structure","language","framework","tool","architecture","metric","person","organization"]

<Input Text>
```
Laravel là một PHP framework phổ biến triển khai mô hình kiến trúc MVC. Nó cung cấp routing và Eloquent ORM. Để chạy Laravel, máy cần PHP và web server như Apache hoặc Nginx.
```

<Output>
entity{tuple_delimiter}Laravel{tuple_delimiter}framework{tuple_delimiter}Laravel là PHP framework phổ biến, cung cấp routing và Eloquent ORM, triển khai mô hình MVC.
entity{tuple_delimiter}PHP{tuple_delimiter}language{tuple_delimiter}PHP là ngôn ngữ lập trình server-side, là dependency cốt lõi của Laravel.
entity{tuple_delimiter}MVC{tuple_delimiter}architecture{tuple_delimiter}MVC (Model-View-Controller) là mô hình kiến trúc phần mềm, được Laravel triển khai.
entity{tuple_delimiter}Eloquent ORM{tuple_delimiter}tool{tuple_delimiter}Eloquent ORM là object-relational mapper tích hợp trong Laravel, hỗ trợ thao tác cơ sở dữ liệu.
entity{tuple_delimiter}Apache{tuple_delimiter}tool{tuple_delimiter}Apache là web server phổ biến dùng để chạy ứng dụng PHP/Laravel.
entity{tuple_delimiter}Nginx{tuple_delimiter}tool{tuple_delimiter}Nginx là web server hiệu năng cao, có thể thay thế Apache cho ứng dụng PHP/Laravel.
relation{tuple_delimiter}Laravel{tuple_delimiter}PHP{tuple_delimiter}language dependency, requires{tuple_delimiter}Laravel sử dụng PHP làm ngôn ngữ nền tảng.
relation{tuple_delimiter}Laravel{tuple_delimiter}MVC{tuple_delimiter}architecture pattern, implements{tuple_delimiter}Laravel hiện thực mô hình kiến trúc MVC.
relation{tuple_delimiter}Laravel{tuple_delimiter}Eloquent ORM{tuple_delimiter}tool dependency, uses{tuple_delimiter}Laravel sử dụng Eloquent ORM để thao tác cơ sở dữ liệu.
relation{tuple_delimiter}Laravel{tuple_delimiter}Apache{tuple_delimiter}runtime environment, hosted by{tuple_delimiter}Laravel có thể chạy trên web server Apache.
relation{tuple_delimiter}Laravel{tuple_delimiter}Nginx{tuple_delimiter}runtime environment, hosted by{tuple_delimiter}Laravel có thể chạy trên web server Nginx.
{completion_delimiter}

""",
    """<Entity_types>
["concept","algorithm","data_structure","language","framework","tool","architecture","metric","person","organization"]

<Input Text>
```
Thuật toán Dijkstra do Edsger Dijkstra phát triển năm 1956, dùng để tìm đường đi ngắn nhất trên đồ thị có trọng số không âm. Thuật toán sử dụng priority queue để chọn đỉnh có khoảng cách nhỏ nhất tại mỗi bước. Khác với Bellman-Ford, Dijkstra không xử lý được trọng số âm. Độ phức tạp thời gian là O((V+E)logV) khi dùng binary heap.
```

<Output>
entity{tuple_delimiter}Dijkstra Algorithm{tuple_delimiter}algorithm{tuple_delimiter}Dijkstra Algorithm là thuật toán tìm đường đi ngắn nhất trên đồ thị có trọng số không âm, do Edsger Dijkstra phát triển năm 1956.
entity{tuple_delimiter}Edsger Dijkstra{tuple_delimiter}person{tuple_delimiter}Edsger Dijkstra là nhà khoa học máy tính, người phát triển thuật toán Dijkstra năm 1956.
entity{tuple_delimiter}Shortest Path{tuple_delimiter}concept{tuple_delimiter}Shortest Path là bài toán tìm đường đi ngắn nhất giữa các đỉnh trên đồ thị, được giải bằng các thuật toán như Dijkstra hoặc Bellman-Ford.
entity{tuple_delimiter}Priority Queue{tuple_delimiter}data_structure{tuple_delimiter}Priority Queue là cấu trúc dữ liệu hàng đợi ưu tiên, dùng để chọn phần tử có giá trị nhỏ nhất hoặc lớn nhất.
entity{tuple_delimiter}Bellman-Ford Algorithm{tuple_delimiter}algorithm{tuple_delimiter}Bellman-Ford là thuật toán tìm đường đi ngắn nhất xử lý được trọng số âm, khác với Dijkstra.
entity{tuple_delimiter}Binary Heap{tuple_delimiter}data_structure{tuple_delimiter}Binary Heap là cấu trúc dữ liệu cây nhị phân được dùng để hiện thực Priority Queue.
entity{tuple_delimiter}O((V+E)logV){tuple_delimiter}metric{tuple_delimiter}O((V+E)logV) là độ phức tạp thời gian của thuật toán Dijkstra khi dùng binary heap.
relation{tuple_delimiter}Dijkstra Algorithm{tuple_delimiter}Edsger Dijkstra{tuple_delimiter}inventor, developed by{tuple_delimiter}Thuật toán Dijkstra được Edsger Dijkstra phát triển năm 1956.
relation{tuple_delimiter}Dijkstra Algorithm{tuple_delimiter}Shortest Path{tuple_delimiter}solves problem, applied to{tuple_delimiter}Dijkstra Algorithm giải bài toán Shortest Path trên đồ thị có trọng số không âm.
relation{tuple_delimiter}Dijkstra Algorithm{tuple_delimiter}Priority Queue{tuple_delimiter}data structure dependency, uses{tuple_delimiter}Dijkstra sử dụng Priority Queue để chọn đỉnh có khoảng cách nhỏ nhất tại mỗi bước.
relation{tuple_delimiter}Dijkstra Algorithm{tuple_delimiter}Bellman-Ford Algorithm{tuple_delimiter}algorithm comparison, contrasts with{tuple_delimiter}Dijkstra nhanh hơn Bellman-Ford nhưng không xử lý trọng số âm.
relation{tuple_delimiter}Priority Queue{tuple_delimiter}Binary Heap{tuple_delimiter}implementation, implemented by{tuple_delimiter}Priority Queue thường được hiện thực bằng Binary Heap.
relation{tuple_delimiter}Dijkstra Algorithm{tuple_delimiter}O((V+E)logV){tuple_delimiter}complexity, measured by{tuple_delimiter}Độ phức tạp thời gian của Dijkstra khi dùng binary heap là O((V+E)logV).
{completion_delimiter}

""",
    """<Entity_types>
["concept","algorithm","data_structure","language","framework","tool","architecture","metric","person","organization"]

<Input Text>
```
Kiến trúc Von Neumann là mô hình máy tính cơ bản với bộ nhớ chung lưu cả lệnh và dữ liệu. Khác với kiến trúc Harvard có bộ nhớ riêng cho lệnh và dữ liệu, Von Neumann đơn giản hơn nhưng có hạn chế "Von Neumann bottleneck". CPU truy cập bộ nhớ qua bus hệ thống. Cache là bộ nhớ tốc độ cao đặt giữa CPU và bộ nhớ chính để tăng hiệu năng, đo bằng cache hit rate.
```

<Output>
entity{tuple_delimiter}Von Neumann Architecture{tuple_delimiter}architecture{tuple_delimiter}Von Neumann Architecture là mô hình máy tính cơ bản với bộ nhớ chung lưu cả lệnh và dữ liệu, có hạn chế Von Neumann bottleneck.
entity{tuple_delimiter}Harvard Architecture{tuple_delimiter}architecture{tuple_delimiter}Harvard Architecture là mô hình máy tính có bộ nhớ riêng cho lệnh và dữ liệu, khác với Von Neumann.
entity{tuple_delimiter}CPU{tuple_delimiter}concept{tuple_delimiter}CPU là bộ xử lý trung tâm của máy tính, truy cập bộ nhớ qua bus hệ thống.
entity{tuple_delimiter}Cache{tuple_delimiter}concept{tuple_delimiter}Cache là bộ nhớ tốc độ cao đặt giữa CPU và bộ nhớ chính, dùng để tăng hiệu năng truy cập dữ liệu.
entity{tuple_delimiter}System Bus{tuple_delimiter}concept{tuple_delimiter}System Bus là kênh truyền dữ liệu giữa CPU và bộ nhớ chính.
entity{tuple_delimiter}Cache Hit Rate{tuple_delimiter}metric{tuple_delimiter}Cache Hit Rate là tỷ lệ truy cập trúng cache, đo lường hiệu năng của hệ thống cache.
relation{tuple_delimiter}Von Neumann Architecture{tuple_delimiter}Harvard Architecture{tuple_delimiter}architecture comparison, contrasts with{tuple_delimiter}Von Neumann có bộ nhớ chung trong khi Harvard có bộ nhớ riêng cho lệnh và dữ liệu.
relation{tuple_delimiter}CPU{tuple_delimiter}Cache{tuple_delimiter}memory hierarchy, uses{tuple_delimiter}CPU sử dụng Cache để giảm thời gian truy cập bộ nhớ chính.
relation{tuple_delimiter}CPU{tuple_delimiter}System Bus{tuple_delimiter}data path, connected via{tuple_delimiter}CPU truy cập bộ nhớ qua System Bus.
relation{tuple_delimiter}Cache{tuple_delimiter}Cache Hit Rate{tuple_delimiter}performance metric, measured by{tuple_delimiter}Hiệu năng của Cache được đo bằng Cache Hit Rate.
{completion_delimiter}

""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. Input Format: The description list is provided in JSON format. Each JSON object (representing a single description) appears on a new line within the `Description List` section.
2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting or extraneous comments before or after the summary.
3. Comprehensiveness: The summary must integrate all key information from *every* provided description, using ONLY the information provided below. Do NOT add any facts, context, or background knowledge not present in the descriptions. Do not omit any important facts or details.
4. Context: Ensure the summary is written from an objective, third-person perspective; explicitly mention the name of the entity or relation for full clarity and context.
5. Context & Objectivity:
  - Write the summary from an objective, third-person perspective.
  - Explicitly mention the full name of the entity or relation at the beginning of the summary to ensure immediate clarity and context.
6. Conflict Handling:
  - In cases of conflicting or inconsistent descriptions, first determine if these conflicts arise from multiple, distinct entities or relationships that share the same name.
  - If distinct entities/relations are identified, summarize each one *separately* within the overall output.
  - If conflicts within a single entity/relation (e.g., historical discrepancies) exist, attempt to reconcile them or present both viewpoints with noted uncertainty.
7. Length Constraint:The summary's total length must not exceed {summary_length} tokens, while still maintaining depth and completeness.
8. Language: The entire output must be written in {language}. Proper nouns (e.g., personal names, place names, organization names) may in their original language if proper translation is not available.
  - The entire output must be written in {language}.
  - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

---Input---
{description_type} Name: {description_name}

Description List:

```
{description_list}
```

---Output---
"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS[
    "rag_response"
] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.
  - Provide a DETAILED and COMPREHENSIVE answer. Each key concept MUST be explained thoroughly with definitions, examples, and relationships to other concepts.
  - Use ALL relevant information from the context. Do not skip or summarize important details.
  - Aim for a thorough response of at least 500 words when the context provides sufficient information.

3. Formatting & Language:
  - The response MUST always be in Vietnamese.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. Image Markers:
  - The context may contain image markers in the format `[IMG_docname_N]` (e.g., `[IMG_KienTrucMayTinh_CH01_0]`).
  - You MUST preserve these markers exactly as they appear in your response, placing them at the appropriate position in your answer.
  - Do NOT remove, rename, translate, or modify any `[IMG_...]` marker.

5. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

6. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

7. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""

PROMPTS[
    "naive_rag_response"
] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a **References** section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.
  - Provide a DETAILED and COMPREHENSIVE answer. Each key concept MUST be explained thoroughly with definitions, examples, and relationships to other concepts.
  - Use ALL relevant information from the context. Do not skip or summarize important details.
  - Aim for a thorough response of at least 500 words when the context provides sufficient information.

3. Formatting & Language:
  - The response MUST always be in Vietnamese.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. Image Markers:
  - The context may contain image markers in the format `[IMG_docname_N]` (e.g., `[IMG_KienTrucMayTinh_CH01_0]`).
  - You MUST preserve these markers exactly as they appear in your response, placing them at the appropriate position in your answer.
  - Do NOT remove, rename, translate, or modify any `[IMG_...]` marker.

5. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

6. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

7. Additional Instructions: {user_prompt}


---Context---

{content_data}
"""

PROMPTS[
    "kg_query_context"
] = """
Knowledge Graph Data (Entity):

```json
{entities_str}
```

Knowledge Graph Data (Relationship):

```json
{relations_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS[
    "naive_query_context"
] = """
Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS[
    "keywords_extraction"
] = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.
5. **Language**: All extracted keywords MUST be in {language}. Proper nouns (e.g., personal names, place names, organization names) should be kept in their original language.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"

Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}

""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}

""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"

Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}

""",
]
