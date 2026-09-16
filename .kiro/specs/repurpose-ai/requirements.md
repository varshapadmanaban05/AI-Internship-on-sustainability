# Requirements Document

## Introduction

RePurpose AI is a conversational, AI-powered circular reuse and repurposing advisor built as a student internship prototype aligned with SDG 12 (Responsible Consumption & Production). The user describes an unwanted item in natural language, and the system recommends the most appropriate circular pathway from a 7-level hierarchy (Reuse → Repair → Repurpose → Donate → Refurbish → Recycle → Responsible Disposal). Recommendations are grounded in a curated sustainability knowledge base via RAG (Retrieval-Augmented Generation), ensuring advice is cited and traceable. The interface is a Streamlit web app.

---

## Glossary

- **Advisor**: The RePurpose AI system as a whole — the Streamlit app, LLM pipeline, and RAG retrieval engine.
- **User**: The student, intern, or end-user interacting with the Advisor via the Streamlit UI.
- **Item Description**: A natural-language text input submitted by the User describing an unwanted item.
- **Circular Pathway**: One of the seven ranked disposal/reuse strategies: Reuse As-Is, Repair, Repurpose, Donate, Refurbish, Recycle, Responsible Disposal.
- **Knowledge Base**: A curated collection of 30–50 PDFs and web documents covering circular economy frameworks, waste classification, material guides, e-waste, repurposing inspiration, and donation networks.
- **RAG Pipeline**: The Retrieval-Augmented Generation pipeline consisting of the Retriever and the LLM Generator.
- **Retriever**: The vector-search component (ChromaDB or FAISS + sentence-transformers) that fetches relevant chunks from the Knowledge Base.
- **Generator**: The LLM (IBM watsonx.ai or OpenAI API) that produces the final recommendation using retrieved context.
- **Recommendation**: The Advisor's structured output containing: Circular Pathway, Reasoning, Source Citation, Concrete Next Steps, and a Confidence Note.
- **Hazardous Item**: Any item that poses environmental or health risk during disposal, including e-waste, batteries, and chemical products.
- **Safety Warning**: A prominently displayed alert notifying the User of hazardous handling requirements.
- **Conversation History**: The sequence of prior User messages and Advisor responses maintained within a single session.
- **Chunk**: A 300–500 token segment of a Knowledge Base document, stored with metadata (source title, URL or filename, circular economy topic tag).

---

## Requirements

### Requirement 1: Item Description Input

**User Story:** As a User, I want to describe an unwanted item in plain text, so that the Advisor can understand what I have and recommend a circular pathway.

#### Acceptance Criteria

1. THE Advisor SHALL provide a text input field on the main interface where the User can type an Item Description.
2. WHEN the User submits an Item Description of fewer than 5 characters, THE Advisor SHALL display an inline validation message prompting the User to provide a more detailed description.
3. WHEN the User submits an Item Description of 5 or more characters, THE Advisor SHALL accept the input, display a processing indicator, and initiate the RAG Pipeline.
4. THE Advisor SHALL accept Item Descriptions of up to 1000 characters in length.
5. WHEN the User attempts to enter more than 1000 characters into the input field, THE Advisor SHALL prevent input beyond 1000 characters and display a notice at the character limit indicating the maximum has been reached.
6. WHEN an Item Description that was truncated to 1000 characters is submitted, THE Advisor SHALL proceed to initiate the RAG Pipeline using the truncated text.

---

### Requirement 2: RAG-Powered Knowledge Retrieval

**User Story:** As a User, I want the Advisor's recommendations to be grounded in real sustainability references, so that I can trust the advice is accurate and not hallucinated.

#### Acceptance Criteria

1. WHEN the RAG Pipeline is initiated, THE Retriever SHALL convert the Item Description into a vector embedding using a sentence-transformers model, provided the Item Description is between 1 and 500 characters in length.
2. WHEN an embedding is produced, THE Retriever SHALL perform a similarity search against the Knowledge Base and return the top 3 to 5 most relevant Chunks with a cosine similarity score of 0.50 or above.
3. WHEN an embedding is produced and Chunks are retrieved, THE Retriever SHALL attach metadata (source title and filename or URL) to each retrieved Chunk before passing it to the Generator.
4. IF fewer than 3 Chunks with a cosine similarity score of 0.50 or above are found, THEN THE Advisor SHALL proceed with all available Chunks and include a Confidence Note in the response indicating that limited reference material was found; IF zero Chunks meet the threshold, THEN THE Advisor SHALL return a response stating that no relevant references were found and no recommendation can be generated.
5. THE Knowledge Base SHALL contain a minimum of 30 documents covering at least the following topics: circular economy frameworks, material recycling guides, e-waste handling, donation networks, and responsible disposal.
6. THE Retriever SHALL produce a non-empty source title and a non-empty filename or URL for the metadata of every retrieved Chunk passed to the Generator.
7. IF the sentence-transformers model is unavailable or returns an error during embedding, THEN THE Retriever SHALL halt the RAG Pipeline and return an error response indicating that the retrieval service is currently unavailable, without returning a partial or hallucinated recommendation.

---

### Requirement 3: Structured Recommendation Output

**User Story:** As a User, I want the Advisor to give me a clear, structured recommendation with reasoning and next steps, so that I know exactly what to do with my item.

#### Acceptance Criteria

1. WHEN the Generator receives the Item Description and retrieved Chunks, THE Generator SHALL produce a Recommendation containing all five components: Circular Pathway, Reasoning, Source Citation, Concrete Next Steps, and Confidence Note.
2. THE Advisor SHALL display the Circular Pathway as the first element of the Recommendation, visually distinguished from the remaining components.
3. THE Advisor SHALL display Source Citations as a numbered list referencing the source titles of the retrieved Chunks used to produce the Recommendation, with one entry per Chunk.
4. WHEN a Recommendation is generated, THE Advisor SHALL include at least one Concrete Next Step that the User can act on immediately (for example: a local drop-off type, a repair action, or a repurposing project idea).
5. THE Advisor SHALL display the Confidence Note as the last element of the Recommendation, stating whether knowledge base coverage for the described item is high, moderate, or low based on the relevance scores of the retrieved Chunks.
6. WHEN the Generator produces a Recommendation, THE Advisor SHALL present the Recommendation within 30 seconds of the User submitting the Item Description under normal operating conditions.
7. IF the Generator fails to produce a Recommendation within 30 seconds, THEN THE Advisor SHALL display an error message indicating that the request could not be completed and prompt the User to resubmit the Item Description.
8. IF the Generator receives an Item Description but no Chunks are retrieved, THEN THE Generator SHALL produce a Recommendation containing a Confidence Note indicating low knowledge base coverage, with the Circular Pathway and Reasoning based solely on the Item Description.

---

### Requirement 4: 7-Level Circular Pathway Hierarchy

**User Story:** As a User, I want the Advisor to recommend the highest-value circular option for my item, so that waste is minimised and resources are kept in use as long as possible.

#### Acceptance Criteria

1. THE Advisor SHALL rank Circular Pathway recommendations according to the following priority order (highest to lowest): Reuse As-Is, Repair, Repurpose, Donate, Refurbish, Recycle, Responsible Disposal.
2. WHEN the Generator recommends a Circular Pathway that is not Reuse As-Is, THE Advisor SHALL include in the Reasoning an explicit statement of which higher-priority pathways were considered and why each was not applicable to the described item.
3. IF the item described is assessed as functional and in acceptable condition for its original purpose, THEN THE Generator SHALL recommend Reuse As-Is or Donate before Recycle or Responsible Disposal.
4. IF the item described is assessed as non-functional or having specific damage, THEN THE Generator SHALL recommend Repair or Repurpose before Recycle or Responsible Disposal.
5. THE Generator SHALL derive its assessment of item condition from explicit condition indicators in the Item Description (for example: "broken", "cracked", "working", "old but functional") or, in the absence of explicit indicators, SHALL state in the Confidence Note that condition was assumed and prompt the User to confirm.

---

### Requirement 5: Hazardous Item Safety Warning

**User Story:** As a User, I want to be warned immediately if my item requires special handling, so that I do not accidentally cause environmental or health harm.

#### Acceptance Criteria

1. WHEN the Item Description contains one or more keywords associated with Hazardous Items — including but not limited to: battery, batteries, e-waste, electronic, phone, laptop, computer, TV, monitor, chemical, paint, solvent, fluorescent, CFL, LED bulb, mercury, refrigerant, aerosol — THE Advisor SHALL display a Safety Warning before the Recommendation body.
2. THE Safety Warning SHALL contain: (a) a statement that the item requires specialist disposal, and (b) at least one specific guidance point instructing the User on a safe disposal channel (for example: "Do not place in general waste — take to a certified e-waste collection point or manufacturer take-back scheme").
3. WHEN a Safety Warning is displayed, THE Advisor SHALL include as one of the Concrete Next Steps a reference to Responsible Disposal or a certified recycling pathway appropriate to the item category.
4. THE Safety Warning SHALL be rendered in a visually distinct UI element that uses a background colour or border that is different from the Recommendation body, so that a sighted user can identify it as a warning without reading the text.

---

### Requirement 6: Conversational Multi-Turn Interaction

**User Story:** As a User, I want to ask follow-up questions or refine my item description across multiple turns, so that I can get more accurate advice without starting over.

#### Acceptance Criteria

1. THE Advisor SHALL maintain Conversation History from the moment the application loads until the browser tab is closed or the User explicitly clears it.
2. WHEN the User submits a follow-up message, THE Generator SHALL include the most recent 5 exchanges from the Conversation History as context when generating the next Recommendation.
3. WHEN the User clicks the "Clear Conversation" button, THE Advisor SHALL reset the Conversation History to empty and clear all messages from the chat display within the same page interaction.
4. THE Advisor SHALL display Conversation History messages in the UI as a scrollable chat log in chronological order, with User messages and Advisor Recommendations visually differentiated (for example: different alignment or background colour).
5. WHEN the Conversation History exceeds 5 exchanges, THE Advisor SHALL pass only the most recent 5 exchanges to the Generator to stay within LLM context limits, while continuing to display the full history in the UI.

---

### Requirement 7: Streamlit Web Interface

**User Story:** As a User, I want a simple, usable web interface, so that I can interact with the Advisor without needing technical knowledge.

#### Acceptance Criteria

1. THE Advisor SHALL be implemented as a Streamlit application that launches and is accessible via a local web browser at http://localhost:8501 by default.
2. THE Advisor SHALL display on the main page: (a) a title identifying the application, (b) a description of the tool's purpose of no more than 100 words, and (c) a short instruction explaining how to enter an item description.
3. THE Advisor SHALL provide a "Clear Conversation" button that, when clicked, resets the Conversation History and removes all messages from the chat display.
4. WHILE the RAG Pipeline is processing a request, THE Advisor SHALL display a spinner or loading indicator and SHALL NOT allow the User to submit another Item Description until the current request completes.
5. THE Advisor SHALL render the Recommendation using Streamlit markdown, including at minimum: a bold or heading-formatted Circular Pathway label, a bulleted or numbered list for Concrete Next Steps, and a numbered list for Source Citations.
6. THE Advisor SHALL be usable on a standard desktop or laptop browser (Chrome, Firefox, or Edge, current major version) without requiring any client-side installation beyond opening the app URL.

---

### Requirement 8: Knowledge Base Ingestion

**User Story:** As a developer, I want a repeatable script that loads documents into the vector store, so that the Knowledge Base can be updated without rebuilding the app.

#### Acceptance Criteria

1. THE Advisor SHALL include a standalone ingestion script that can be executed independently of the Streamlit app, which reads PDF and plain-text (.txt) documents from a configurable local folder path and loads them into the vector store.
2. WHEN the ingestion script processes a document, THE script SHALL split the document into Chunks of 300 to 500 tokens with a 50-token overlap before embedding.
3. WHEN a Chunk is stored in the vector store, THE script SHALL attach to it metadata containing: (a) the source filename, and (b) a human-readable source title derived from the document filename or a configurable title mapping.
4. WHEN the ingestion script encounters a file it cannot parse (for example: a corrupted PDF or an unsupported file type), THE script SHALL log an error message to the console identifying the filename and the reason for failure, and SHALL continue processing the remaining documents without halting.
5. WHEN the ingestion script completes processing all files in the designated folder, THE script SHALL print to the console a summary line stating the total number of documents successfully processed and the total number of Chunks stored in the vector store.
