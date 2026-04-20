import { useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import { Send, Bot, User, Sparkles, ExternalLink, Loader2 } from 'lucide-react';
import PageHeader from '@/components/PageHeader';
import { askQuestion, listContracts, getContract } from '@/lib/api';
import { cn } from '@/lib/utils';
import type { Contract, ChatMessage } from '@/types';

type RagStrategy = NonNullable<ChatMessage['ragStrategy']>;

const RAG_STRATEGIES: { value: RagStrategy; label: string; description: string }[] = [
  { value: 'adaptive',    label: 'Adaptive',    description: 'Auto-routes by query complexity' },
  { value: 'naive',       label: 'Naive',       description: 'Simple top-K dense retrieval' },
  { value: 'advanced',    label: 'Advanced',    description: 'Hybrid + rerank + compress' },
  { value: 'corrective',  label: 'Corrective',  description: 'NLI gate + web fallback' },
  { value: 'self',        label: 'Self-RAG',    description: 'Reflection tokens' },
  { value: 'graph',       label: 'Graph',       description: 'Knowledge graph traversal' },
];

const SUGGESTED_QUESTIONS = [
  'What is the termination notice period?',
  'Are there any limitation of liability clauses?',
  'What happens if one party breaches the agreement?',
  'What is the governing law?',
];

export default function ChatPage() {
  const { contractId } = useParams();
  const [contract, setContract] = useState<Contract | null>(null);
  const [allContracts, setAllContracts] = useState<Contract[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [strategy, setStrategy] = useState<RagStrategy>('adaptive');
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listContracts().then(setAllContracts);
  }, []);

  useEffect(() => {
    if (contractId) {
      getContract(contractId).then(setContract);
    } else {
      listContracts().then((cs) => {
        if (cs.length > 0) getContract(cs[0].id).then(setContract);
      });
    }
    setMessages([]);
  }, [contractId]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function handleSend(text?: string) {
    const question = text ?? input.trim();
    if (!question || !contract || sending) return;

    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: question,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setSending(true);

    try {
      const reply = await askQuestion(contract.id, question, strategy);
      setMessages((prev) => [...prev, reply]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          id: `e-${Date.now()}`,
          role: 'assistant',
          content: 'Sorry, something went wrong. Please try again.',
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setSending(false);
    }
  }

  return (
    <>
      <PageHeader
        title="Chat with Contract"
        description={contract ? contract.filename : 'Loading contract...'}
      >
        {/* Strategy selector */}
        <select
          value={strategy}
          onChange={(e) => setStrategy(e.target.value as RagStrategy)}
          className="bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-brand-500"
        >
          {RAG_STRATEGIES.map((s) => (
            <option key={s.value} value={s.value}>
              RAG: {s.label}
            </option>
          ))}
        </select>
      </PageHeader>

      <div className="flex h-[calc(100vh-97px)]">
        {/* Contract list sidebar (only if multiple contracts) */}
        {allContracts.length > 1 && (
          <aside className="w-64 border-r border-slate-200 bg-white overflow-y-auto">
            <div className="p-4 text-xs font-semibold uppercase tracking-wide text-slate-400">
              Your Contracts
            </div>
            {allContracts.map((c) => (
              <a
                key={c.id}
                href={`/chat/${c.id}`}
                className={cn(
                  'block px-4 py-3 text-sm border-l-2 transition-colors',
                  c.id === contract?.id
                    ? 'border-brand-600 bg-brand-50 text-brand-900'
                    : 'border-transparent hover:bg-slate-50 text-slate-700'
                )}
              >
                <div className="font-medium truncate">{c.filename}</div>
                <div className="text-xs text-slate-500">{c.pages} pages · {c.jurisdiction}</div>
              </a>
            ))}
          </aside>
        )}

        {/* Main chat area */}
        <div className="flex-1 flex flex-col">
          <div className="flex-1 overflow-y-auto">
            {messages.length === 0 ? (
              <div className="h-full flex items-center justify-center p-8">
                <div className="max-w-lg text-center">
                  <div className="w-14 h-14 rounded-full bg-brand-50 flex items-center justify-center mx-auto mb-4">
                    <Sparkles size={24} className="text-brand-600" />
                  </div>
                  <h2 className="text-xl font-semibold text-slate-900 mb-2">
                    Ask anything about this contract
                  </h2>
                  <p className="text-sm text-slate-500 mb-6">
                    Answers are grounded in contract text with NLI-verified citations.
                  </p>
                  <div className="grid grid-cols-1 gap-2">
                    {SUGGESTED_QUESTIONS.map((q) => (
                      <button
                        key={q}
                        onClick={() => handleSend(q)}
                        className="text-left px-4 py-3 rounded-lg border border-slate-200 hover:border-brand-400 hover:bg-brand-50 text-sm text-slate-700 transition-colors"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="max-w-3xl mx-auto p-6 space-y-6">
                {messages.map((msg) => (
                  <MessageBubble key={msg.id} message={msg} />
                ))}
                {sending && (
                  <div className="flex gap-3 items-start">
                    <div className="w-8 h-8 rounded-full bg-brand-50 flex items-center justify-center flex-shrink-0">
                      <Bot size={16} className="text-brand-600" />
                    </div>
                    <div className="flex items-center gap-2 pt-1.5 text-sm text-slate-500">
                      <Loader2 size={14} className="animate-spin" />
                      Thinking...
                    </div>
                  </div>
                )}
                <div ref={endRef} />
              </div>
            )}
          </div>

          {/* Input */}
          <div className="border-t border-slate-200 bg-white p-4">
            <div className="max-w-3xl mx-auto flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
                placeholder="Ask a question about this contract..."
                disabled={sending || !contract}
                className="flex-1 px-4 py-2.5 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent disabled:opacity-50"
              />
              <button
                onClick={() => handleSend()}
                disabled={!input.trim() || sending || !contract}
                className="btn-primary inline-flex items-center gap-2"
              >
                <Send size={16} />
                Send
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';
  return (
    <div className={cn('flex gap-3 items-start', isUser && 'flex-row-reverse')}>
      <div className={cn(
        'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0',
        isUser ? 'bg-slate-900 text-white' : 'bg-brand-50 text-brand-600'
      )}>
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>
      <div className={cn('flex-1 min-w-0', isUser && 'flex justify-end')}>
        <div className={cn(
          'inline-block max-w-full rounded-2xl px-4 py-3',
          isUser
            ? 'bg-brand-600 text-white'
            : 'bg-white border border-slate-200 text-slate-800'
        )}>
          <div className="prose prose-sm max-w-none prose-p:my-1">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        </div>

        {/* Citations */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-2 space-y-1.5">
            {message.sources.map((source) => (
              <div
                key={source.clauseId}
                className="text-xs bg-slate-50 border border-slate-200 rounded-lg p-3"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-semibold text-slate-700">{source.clauseType}</span>
                  <span className="text-slate-400">
                    Page {source.pageNumber} · {(source.relevanceScore * 100).toFixed(0)}% match
                  </span>
                </div>
                <p className="text-slate-600 italic">"{source.snippet}"</p>
              </div>
            ))}
          </div>
        )}

        {/* Metadata bar */}
        {!isUser && message.confidence !== undefined && (
          <div className="mt-2 flex items-center gap-3 text-xs text-slate-400">
            <span className="flex items-center gap-1">
              <ExternalLink size={11} />
              {message.ragStrategy}
            </span>
            <span>Confidence: {(message.confidence * 100).toFixed(0)}%</span>
            {message.correctionRounds !== undefined && message.correctionRounds > 0 && (
              <span>{message.correctionRounds} self-correction{message.correctionRounds > 1 ? 's' : ''}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
