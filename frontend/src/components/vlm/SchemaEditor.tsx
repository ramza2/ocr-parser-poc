import type { SchemaField } from "../../types/vlm";

interface Props {
  schema: SchemaField[];
  onChange: (schema: SchemaField[]) => void;
  disabled?: boolean;
}

const TYPES = ["text", "number", "date", "phone", "email", "address"];

export default function SchemaEditor({ schema, onChange, disabled }: Props) {
  const addField = () => {
    onChange([...schema, { key: "", description: "", type: "text" }]);
  };

  const removeField = (idx: number) => {
    onChange(schema.filter((_, i) => i !== idx));
  };

  const updateField = (idx: number, patch: Partial<SchemaField>) => {
    onChange(schema.map((f, i) => (i === idx ? { ...f, ...patch } : f)));
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-700">Schema 정의</h3>
        <button
          onClick={addField}
          disabled={disabled}
          className="rounded bg-brand-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-brand-700 disabled:opacity-50"
        >
          + 추가
        </button>
      </div>

      {schema.length === 0 && (
        <p className="text-center text-xs text-slate-400 py-3">
          추출할 항목을 추가하세요.
        </p>
      )}

      <div className="space-y-1.5 max-h-52 overflow-y-auto">
        {schema.map((field, idx) => (
          <div key={idx} className="flex gap-1.5 items-center">
            <input
              value={field.key}
              onChange={(e) => updateField(idx, { key: e.target.value })}
              placeholder="Key"
              disabled={disabled}
              className="w-24 shrink-0 rounded border border-slate-200 px-2 py-1.5 text-xs focus:border-brand-400 focus:outline-none"
            />
            <input
              value={field.description}
              onChange={(e) => updateField(idx, { description: e.target.value })}
              placeholder="설명 (선택)"
              disabled={disabled}
              className="min-w-0 flex-1 rounded border border-slate-200 px-2 py-1.5 text-xs focus:border-brand-400 focus:outline-none"
            />
            <select
              value={field.type}
              onChange={(e) => updateField(idx, { type: e.target.value })}
              disabled={disabled}
              className="w-20 shrink-0 rounded border border-slate-200 px-1 py-1.5 text-xs focus:border-brand-400 focus:outline-none"
            >
              {TYPES.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
            <button
              onClick={() => removeField(idx)}
              disabled={disabled}
              className="shrink-0 rounded p-1 text-slate-400 hover:bg-red-50 hover:text-red-500 disabled:opacity-50"
              title="삭제"
            >
              <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
