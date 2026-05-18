interface Props {
  text: string;
}

export default function TextResultView({ text }: Props) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <h3 className="mb-3 text-sm font-semibold text-slate-800">추출된 텍스트</h3>
      {text ? (
        <pre className="max-h-96 overflow-auto whitespace-pre-wrap rounded-lg bg-slate-50 p-4 font-mono text-xs leading-relaxed text-slate-700">
          {text}
        </pre>
      ) : (
        <p className="text-sm text-slate-500">표시할 텍스트가 없습니다.</p>
      )}
    </section>
  );
}
