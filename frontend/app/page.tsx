"use client";

import { ChangeEvent, DragEvent, FormEvent, useEffect, useRef, useState } from "react";
import {
  Bot,
  Bug,
  CheckCircle2,
  Activity,
  CheckSquare,
  FileText,
  History,
  Loader2,
  MessageSquare,
  Moon,
  PanelRightOpen,
  Send,
  Settings2,
  Sun,
  Trash2,
  UploadCloud,
  User
} from "lucide-react";

type DocumentMetadata = {
  document_id: string;
  filename: string;
  content_type: string;
  pages?: number;
  chunks: number;
  uploaded_at: string;
};

type Citation = {
  document_id: string;
  filename: string;
  page?: number;
  chunk_id: string;
  score?: number;
  excerpt: string;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at?: string;
  citations?: Citation[];
  retrievalDebug?: Record<string, unknown>[];
  qualityEvaluation?: { score?: number; reason?: string };
};

type ObservabilityEvent = {
  created_at: string;
  rewritten_query: string;
  quality_score?: number;
  quality_reason?: string;
  answer_chars: number;
  retrieval_stages: string[];
  top_chunks: Record<string, unknown>[];
};

type ObservabilitySummary = {
  total_queries: number;
  average_quality_score: number;
  latest_events: ObservabilityEvent[];
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api";

function createId() {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

const welcomeMessage: ChatMessage = {
  id: "welcome",
  role: "assistant",
  content: "Upload documents, then ask a grounded question. I will retrieve, rank, and cite the sources I used."
};

export default function Home() {
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([welcomeMessage]);
  const [historyMessages, setHistoryMessages] = useState<ChatMessage[]>([]);
  const [observability, setObservability] = useState<ObservabilitySummary | null>(null);
  const [input, setInput] = useState("");
  const [isUploading, setUploading] = useState(false);
  const [isThinking, setThinking] = useState(false);
  const [isHistoryLoading, setHistoryLoading] = useState(false);
  const [debugMode, setDebugMode] = useState(true);
  const [darkMode, setDarkMode] = useState(true);
  const [activeView, setActiveView] = useState<"chat" | "documents" | "history" | "observability">("chat");
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [sessionId, setSessionId] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const storedSession = window.localStorage.getItem("advancedrag_session_id");
    const nextSession = storedSession || createId();
    window.localStorage.setItem("advancedrag_session_id", nextSession);
    setSessionId(nextSession);
    // Wake up the backend (Render free tier spins down on inactivity)
    fetch(`${API_BASE}/health`).catch(() => {});
    refreshDocuments();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isThinking]);

  async function refreshDocuments() {
    const response = await fetch(`${API_BASE}/documents`);
    if (response.ok) {
      const data = (await response.json()) as DocumentMetadata[];
      setDocuments(data);
      setSelectedDocs((current) => current.filter((id) => data.some((doc) => doc.document_id === id)));
    }
  }

  async function deleteDocument(documentId: string) {
    try {
      const response = await fetch(`${API_BASE}/documents/${documentId}`, { method: "DELETE" });
      if (!response.ok) throw new Error(await response.text());
      setDocuments((current) => current.filter((doc) => doc.document_id !== documentId));
      setSelectedDocs((current) => current.filter((id) => id !== documentId));
    } catch (error) {
      window.alert(
        `Could not delete the document. Make sure the backend is running at ${API_BASE}. ${
          error instanceof Error ? error.message : ""
        }`
      );
    }
  }

  async function deleteDuplicateDocuments(filename: string) {
    const matching = documents
      .filter((doc) => doc.filename === filename)
      .sort((a, b) => new Date(b.uploaded_at).getTime() - new Date(a.uploaded_at).getTime());
    await Promise.all(matching.slice(1).map((doc) => deleteDocument(doc.document_id)));
  }

  async function refreshHistory() {
    if (!sessionId) return;
    setHistoryLoading(true);
    try {
      const response = await fetch(`${API_BASE}/chat/${sessionId}/history`);
      if (!response.ok) throw new Error(await response.text());
      const data = (await response.json()) as Omit<ChatMessage, "id">[];
      setHistoryMessages(data.map((message, index) => ({ ...message, id: `history-${index}` })));
    } finally {
      setHistoryLoading(false);
    }
  }

  async function refreshObservability() {
    const response = await fetch(`${API_BASE}/observability/summary`);
    if (response.ok) {
      setObservability((await response.json()) as ObservabilitySummary);
    }
  }

  async function clearHistory() {
    if (!sessionId) return;
    await fetch(`${API_BASE}/chat/${sessionId}/history`, { method: "DELETE" });
    setHistoryMessages([]);
    setMessages([welcomeMessage]);
  }

  async function uploadFile(file: File) {
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      // First ping to wake up Render free tier if sleeping
      await fetch(`${API_BASE}/health`).catch(() => {});
      const response = await fetch(`${API_BASE}/documents/upload`, { method: "POST", body: formData });
      if (!response.ok) throw new Error(await response.text());
      const doc = (await response.json()) as DocumentMetadata;
      setDocuments((current) => [doc, ...current.filter((item) => item.document_id !== doc.document_id)]);
      setSelectedDocs((current) => [...new Set([...current, doc.document_id])]);
      setActiveView("documents");
    } catch (error) {
      window.alert(
        `Could not upload the file. Make sure the backend is running at ${API_BASE}. ${
          error instanceof Error ? error.message : ""
        }`
      );
    } finally {
      setUploading(false);
    }
  }

  async function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) await uploadFile(file);
  }

  async function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file) await uploadFile(file);
  }

  async function submitMessage(event: FormEvent) {
    event.preventDefault();
    const question = input.trim();
    if (!question || isThinking) return;
    const activeSession = sessionId || createId();
    if (!sessionId) {
      window.localStorage.setItem("advancedrag_session_id", activeSession);
      setSessionId(activeSession);
    }

    setInput("");
    const assistantId = createId();
    setMessages((current) => [
      ...current,
      { id: createId(), role: "user", content: question },
      { id: assistantId, role: "assistant", content: "" }
    ]);
    setThinking(true);

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: activeSession,
          message: question,
          document_ids: selectedDocs.length ? selectedDocs : undefined,
          stream: true,
          debug: true
        })
      });
      if (!response.ok || !response.body) throw new Error(await response.text());
      await readEventStream(response, assistantId);
    } catch (error) {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? { ...message, content: `I could not complete that request. ${error instanceof Error ? error.message : ""}` }
            : message
        )
      );
    } finally {
      setThinking(false);
    }
  }

  async function readEventStream(response: Response, assistantId: string) {
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";
      for (const event of events) {
        const eventType = event.match(/^event: (.+)$/m)?.[1];
        const data = event.match(/^data: (.*)$/m)?.[1] ?? "";
        if (eventType === "metadata") {
          const metadata = JSON.parse(data);
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    citations: metadata.citations,
                    retrievalDebug: metadata.retrieval_debug,
                    qualityEvaluation: metadata.quality_evaluation
                  }
                : message
            )
          );
        }
        if (eventType === "token") {
          const token = JSON.parse(data);
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId ? { ...message, content: `${message.content}${token}` } : message
            )
          );
        }
        if (eventType === "quality") {
          const quality = JSON.parse(data);
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId ? { ...message, qualityEvaluation: quality } : message
            )
          );
        }
      }
    }
  }

  const shellClass = darkMode ? "app dark" : "app light";
  const groupedDocuments = documents
    .reduce<Array<DocumentMetadata & { document_ids: string[]; uploads: number }>>((groups, doc) => {
      const existing = groups.find((item) => item.filename === doc.filename);
      if (!existing) {
        groups.push({ ...doc, document_ids: [doc.document_id], uploads: 1 });
        return groups;
      }
      existing.chunks += doc.chunks;
      existing.document_ids.push(doc.document_id);
      existing.uploads += 1;
      return groups;
    }, [])
    .sort((a, b) => new Date(b.uploaded_at).getTime() - new Date(a.uploaded_at).getTime());

  return (
    <main className={shellClass}>
      <aside className="sidebar">
        <div className="brand">
          <div className="brandMark">AR</div>
          <div>
            <h1>AdvancedRAG AI</h1>
            <p>Source-grounded document intelligence</p>
          </div>
        </div>

        <label className="uploadZone" onDrop={onDrop} onDragOver={(event) => event.preventDefault()}>
          {isUploading ? <Loader2 className="spin" /> : <UploadCloud />}
          <span>{isUploading ? "Indexing document..." : "Drop PDF or document"}</span>
          <input type="file" accept=".pdf,.txt,.md,.csv,.doc,.docx" onChange={onFileChange} />
        </label>

        <section className="panel">
          <div className="panelTitle splitTitle">
            <span>
              <FileText size={16} />
              Documents
            </span>
            <small>{documents.length} indexed</small>
          </div>
          <div className="documentToolbar">
            <button onClick={() => setSelectedDocs(documents.map((doc) => doc.document_id))}>
              <CheckSquare size={14} />
              All
            </button>
            <button onClick={() => setSelectedDocs([])}>Clear</button>
          </div>
          <div className="docList">
            {groupedDocuments.map((doc) => {
              const selectedCount = doc.document_ids.filter((id) => selectedDocs.includes(id)).length;
              return (
                <div key={doc.filename} className={selectedCount ? "docGroup active" : "docGroup"}>
                  <button
                    className="docMain"
                    onClick={() =>
                      setSelectedDocs((current) => {
                        const allSelected = doc.document_ids.every((id) => current.includes(id));
                        return allSelected
                          ? current.filter((id) => !doc.document_ids.includes(id))
                          : [...new Set([...current, ...doc.document_ids])];
                      })
                    }
                  >
                    <FileText size={16} />
                    <span title={doc.filename}>{doc.filename}</span>
                    <small>
                      {doc.chunks} chunks
                      {doc.uploads > 1 ? ` - ${doc.uploads} uploads` : ""}
                    </small>
                  </button>
                  <div className="docActions">
                    {doc.uploads > 1 && (
                      <button title="Remove duplicate uploads" onClick={() => deleteDuplicateDocuments(doc.filename)}>
                        <Trash2 size={14} />
                        Dedupe
                      </button>
                    )}
                    <button title="Delete latest upload" onClick={() => deleteDocument(doc.document_ids[0])}>
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              );
            })}
            {!documents.length && <p className="muted">No documents indexed yet.</p>}
          </div>
        </section>

        <section className="panel compact">
          <div className="panelTitle">
            <Settings2 size={16} />
            <span>Workspace</span>
          </div>
          <button
            className={activeView === "chat" ? "toggleRow activeNav" : "toggleRow"}
            onClick={() => setActiveView("chat")}
          >
            <MessageSquare size={16} />
            <span>Chat</span>
            <b>{activeView === "chat" ? "Open" : "View"}</b>
          </button>
          <button
            className={activeView === "documents" ? "toggleRow activeNav" : "toggleRow"}
            onClick={() => {
              setActiveView("documents");
              refreshDocuments();
            }}
          >
            <FileText size={16} />
            <span>Documents page</span>
            <b>{activeView === "documents" ? "Open" : "View"}</b>
          </button>
          <button
            className={activeView === "history" ? "toggleRow activeNav" : "toggleRow"}
            onClick={() => {
              setActiveView("history");
              refreshHistory();
            }}
          >
            <History size={16} />
            <span>History page</span>
            <b>{activeView === "history" ? "Open" : "View"}</b>
          </button>
          <button
            className={activeView === "observability" ? "toggleRow activeNav" : "toggleRow"}
            onClick={() => {
              setActiveView("observability");
              refreshObservability();
            }}
          >
            <Activity size={16} />
            <span>Observability</span>
            <b>{activeView === "observability" ? "Open" : "View"}</b>
          </button>
          <button className="toggleRow activeNav" onClick={() => setDebugMode(true)}>
            <Bug size={16} />
            <span>Retrieval debug</span>
            <b>Always on</b>
          </button>
          <button className="toggleRow" onClick={() => setDarkMode((value) => !value)}>
            {darkMode ? <Moon size={16} /> : <Sun size={16} />}
            <span>Theme</span>
            <b>{darkMode ? "Dark" : "Light"}</b>
          </button>
        </section>
      </aside>

      <section className="chatShell">
        <header className="topbar">
          <div>
            <span className="eyebrow">
              {activeView === "chat"
                ? "Multi-document RAG"
                : activeView === "documents"
                  ? "Workspace Files"
                  : activeView === "history"
                    ? "Session Memory"
                    : "Pipeline Metrics"}
            </span>
            <h2>
              {activeView === "chat"
                ? "Ask across your knowledge base"
                : activeView === "documents"
                  ? "Document workspace"
                  : activeView === "history"
                    ? "Chat history"
                    : "Observability dashboard"}
            </h2>
          </div>
          {activeView === "chat" ? (
            <div className="statusPill">
              <CheckCircle2 size={16} />
              {selectedDocs.length || documents.length} active sources
            </div>
          ) : activeView === "documents" ? (
            <div className="historyActions">
              <button onClick={refreshDocuments}>
                <FileText size={16} />
                Refresh
              </button>
              <button onClick={() => setSelectedDocs(documents.map((doc) => doc.document_id))}>
                <CheckSquare size={16} />
                Select all
              </button>
            </div>
          ) : activeView === "history" ? (
            <div className="historyActions">
              <button onClick={refreshHistory} disabled={isHistoryLoading}>
                {isHistoryLoading ? <Loader2 className="spin" size={16} /> : <History size={16} />}
                Refresh
              </button>
              <button onClick={clearHistory}>
                <MessageSquare size={16} />
                Clear
              </button>
            </div>
          ) : (
            <div className="historyActions">
              <button onClick={refreshObservability}>
                <Activity size={16} />
                Refresh
              </button>
            </div>
          )}
        </header>

        {activeView === "chat" ? (
          <div className="chatStream">
            {messages.map((message) => (
              <article key={message.id} className={`message ${message.role}`}>
                <div className="avatar">{message.role === "assistant" ? <Bot size={18} /> : <User size={18} />}</div>
                <div className="bubble">
                  <p>{message.content || "Thinking through retrieved context..."}</p>
                  {!!message.citations?.length && (
                    <div className="citations">
                      {message.citations.map((citation) => (
                        <button key={citation.chunk_id} onClick={() => setActiveCitation(citation)}>
                          {citation.filename} {citation.page ? `p.${citation.page}` : ""} -{" "}
                          {citation.score ? citation.score.toFixed(2) : "ranked"}
                        </button>
                      ))}
                    </div>
                  )}
                  {debugMode && message.role === "assistant" && !!message.retrievalDebug?.length && (
                    <details className="debugPanel">
                      <summary>Retrieval debug</summary>
                      <div className="debugGrid">
                        {message.retrievalDebug.slice(0, 8).map((row, index) => (
                          <div key={`${message.id}-${index}`} className="debugRow">
                            <span>{String(row.stage ?? "vector")}</span>
                            <b>{String(row.chunk_id ?? "chunk")}</b>
                            <small>
                              {row.score !== undefined ? `score ${Number(row.score).toFixed(2)}` : ""}
                              {row.llm_score !== undefined ? ` rerank ${Number(row.llm_score).toFixed(2)}` : ""}
                            </small>
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                  {message.role === "assistant" && message.qualityEvaluation?.reason && (
                    <div className="quality">
                      Quality {message.qualityEvaluation.score ?? 0}/10 - {message.qualityEvaluation.reason}
                    </div>
                  )}
                </div>
              </article>
            ))}
            {isThinking && (
              <div className="typing">
                <Loader2 className="spin" size={16} />
                Retrieving, reranking, and grounding the answer
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        ) : activeView === "documents" ? (
          <div className="historyPage">
            <div className="metricGrid">
              <div className="metricCard">
                <span>Indexed files</span>
                <b>{documents.length}</b>
              </div>
              <div className="metricCard">
                <span>Selected</span>
                <b>{selectedDocs.length}</b>
              </div>
              <div className="metricCard">
                <span>Total chunks</span>
                <b>{documents.reduce((total, doc) => total + doc.chunks, 0)}</b>
              </div>
            </div>
            <div className="documentTable">
              {documents.map((doc) => (
                <article key={doc.document_id} className={selectedDocs.includes(doc.document_id) ? "documentRow active" : "documentRow"}>
                  <button
                    className="documentRowMain"
                    onClick={() =>
                      setSelectedDocs((current) =>
                        current.includes(doc.document_id)
                          ? current.filter((id) => id !== doc.document_id)
                          : [...current, doc.document_id]
                      )
                    }
                  >
                    <FileText size={18} />
                    <span>{doc.filename}</span>
                    <small>
                      {doc.chunks} chunks - uploaded {new Date(doc.uploaded_at).toLocaleString()}
                    </small>
                  </button>
                  <button className="dangerButton" onClick={() => deleteDocument(doc.document_id)}>
                    <Trash2 size={15} />
                    Delete
                  </button>
                </article>
              ))}
              {!documents.length && (
                <div className="emptyState">
                  <FileText size={28} />
                  <p>No documents uploaded yet. Drop a file in the sidebar and it will appear here at the top.</p>
                </div>
              )}
            </div>
          </div>
        ) : activeView === "history" ? (
          <div className="historyPage">
            {historyMessages.map((message) => (
              <article key={message.id} className="historyItem">
                <div className="historyMeta">
                  <span>{message.role}</span>
                  {message.created_at && <time>{new Date(message.created_at).toLocaleString()}</time>}
                </div>
                <p>{message.content}</p>
                {!!message.citations?.length && (
                  <div className="citations">
                    {message.citations.map((citation) => (
                      <button key={citation.chunk_id} onClick={() => setActiveCitation(citation)}>
                        {citation.filename} {citation.page ? `p.${citation.page}` : ""}
                      </button>
                    ))}
                  </div>
                )}
              </article>
            ))}
            {!historyMessages.length && (
              <div className="emptyState">
                <History size={28} />
                <p>No saved chat history yet. Ask a question in the chat view, then come back here.</p>
              </div>
            )}
          </div>
        ) : (
          <div className="historyPage">
            <div className="metricGrid">
              <div className="metricCard">
                <span>Total queries</span>
                <b>{observability?.total_queries ?? 0}</b>
              </div>
              <div className="metricCard">
                <span>Avg self-check</span>
                <b>{observability?.average_quality_score ?? 0}/10</b>
              </div>
              <div className="metricCard">
                <span>Retrieval mode</span>
                <b>Hybrid</b>
              </div>
            </div>
            {observability?.latest_events.map((event, index) => (
              <article key={`${event.created_at}-${index}`} className="historyItem">
                <div className="historyMeta">
                  <span>query</span>
                  <time>{new Date(event.created_at).toLocaleString()}</time>
                </div>
                <p>{event.rewritten_query}</p>
                <div className="quality">
                  Self-check {event.quality_score ?? 0}/10 - {event.quality_reason ?? "No reason recorded"}
                </div>
                <div className="stageList">
                  {event.retrieval_stages.map((stage) => (
                    <span key={stage}>{stage}</span>
                  ))}
                </div>
                <details className="debugPanel">
                  <summary>Top chunks</summary>
                  <div className="debugGrid">
                    {event.top_chunks.map((row, rowIndex) => (
                      <div key={rowIndex} className="debugRow">
                        <span>{String(row.stage ?? "stage")}</span>
                        <b>{String(row.chunk_id ?? "chunk")}</b>
                        <small>{row.score !== undefined ? `score ${Number(row.score).toFixed(2)}` : ""}</small>
                      </div>
                    ))}
                  </div>
                </details>
              </article>
            ))}
            {!observability?.latest_events.length && (
              <div className="emptyState">
                <Activity size={28} />
                <p>No pipeline metrics yet. Ask a document question, then refresh this dashboard.</p>
              </div>
            )}
          </div>
        )}

        <form className={activeView === "chat" ? "composer" : "composer hiddenComposer"} onSubmit={submitMessage}>
          <History size={18} />
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask about uploaded contracts, papers, notes, policies..."
          />
          <button aria-label="Send message" disabled={isThinking || !input.trim()}>
            <Send size={18} />
          </button>
        </form>
      </section>

      <aside className="sourceViewer">
        <div className="panelTitle">
          <PanelRightOpen size={16} />
          <span>Source Viewer</span>
        </div>
        {activeCitation ? (
          <div className="sourceCard">
            <h3>{activeCitation.filename}</h3>
            <p className="muted">Page {activeCitation.page ?? "unknown"} - {activeCitation.chunk_id}</p>
            <blockquote>{activeCitation.excerpt}</blockquote>
          </div>
        ) : (
          <p className="muted">Select a citation after an answer to inspect the supporting passage.</p>
        )}
      </aside>
    </main>
  );
}
