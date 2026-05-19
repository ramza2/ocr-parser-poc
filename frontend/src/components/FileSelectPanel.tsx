import type { SelectedFileInfo } from "../types/parser";
import { formatFileSize } from "../utils/fileUtils";

interface Props {
  selectedFile: SelectedFileInfo | null;
  onSelect: (file: File) => void;
  disabled?: boolean;
}

export default function FileSelectPanel({
  selectedFile,
  onSelect,
  disabled,
}: Props) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onSelect(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (disabled) return;
    const file = e.dataTransfer.files?.[0];
    if (file) onSelect(file);
  };

  return (
    <section className="space-y-3">
      <h2 className="text-sm font-semibold text-slate-800">1. 파일 선택</h2>

      <label
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-4 py-8 text-center transition ${
          disabled
            ? "cursor-not-allowed border-slate-200 bg-slate-50 opacity-60"
            : "border-slate-300 bg-white hover:border-brand-500 hover:bg-brand-50/30"
        }`}
      >
        <input
          type="file"
          accept=".pdf,.jpg,.jpeg,.png,.tif,.tiff"
          className="hidden"
          disabled={disabled}
          onChange={handleChange}
        />
        <div className="mb-2 text-2xl text-slate-400">↑</div>
        <p className="text-sm font-medium text-slate-700">
          파일을 드래그하거나 클릭하여 업로드
        </p>
        <p className="mt-1 text-xs text-slate-500">
          스캔 PDF, JPG, PNG 지원 (최대 50MB)
        </p>
      </label>

      {selectedFile ? (
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
          <p className="truncate text-sm font-medium text-slate-800">
            {selectedFile.name}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            {selectedFile.extension.toUpperCase()} ·{" "}
            {formatFileSize(selectedFile.size)} · {selectedFile.mimeType}
          </p>
        </div>
      ) : (
        <p className="text-xs text-slate-500">선택된 파일이 없습니다.</p>
      )}
    </section>
  );
}
