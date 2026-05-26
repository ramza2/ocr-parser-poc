interface Props {
  value: number | null | undefined;
}

export default function ConfidenceBadge({ value }: Props) {
  if (value == null) return null;

  const pct = Math.round(value * 100);
  let color = "bg-emerald-100 text-emerald-700";
  if (pct < 70) color = "bg-rose-100 text-rose-700";
  else if (pct < 90) color = "bg-amber-100 text-amber-700";

  return (
    <span className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-semibold ${color}`}>
      {pct}%
    </span>
  );
}
