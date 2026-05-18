interface Props {
  onRun: () => void;
  onReset: () => void;
  canRun: boolean;
  isRunning: boolean;
}

export default function RunControlPanel({
  onRun,
  onReset,
  canRun,
  isRunning,
}: Props) {
  return (
    <div className="flex gap-2 pt-2">
      <button
        type="button"
        onClick={onRun}
        disabled={!canRun || isRunning}
        className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-brand-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
      >
        {isRunning ? (
          <>
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
            처리 중...
          </>
        ) : (
          <>▶ 파서 실행</>
        )}
      </button>
      <button
        type="button"
        onClick={onReset}
        disabled={isRunning}
        className="rounded-lg border border-slate-300 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:opacity-50"
      >
        초기화
      </button>
    </div>
  );
}
