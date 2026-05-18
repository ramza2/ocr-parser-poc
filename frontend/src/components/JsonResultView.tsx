interface Props {
  data: unknown;
}

export default function JsonResultView({ data }: Props) {
  const json = JSON.stringify(data, null, 2);

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <h3 className="mb-3 text-sm font-semibold text-slate-800">Raw JSON Data</h3>
      <pre className="max-h-96 overflow-auto rounded-lg bg-slate-900 p-4 font-mono text-xs leading-relaxed text-emerald-300">
        {json}
      </pre>
    </section>
  );
}
