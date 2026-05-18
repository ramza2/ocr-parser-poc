const SUPPORTED = new Set(["pdf", "jpg", "jpeg", "png", "tif", "tiff"]);

export function getExtension(filename: string): string {
  const parts = filename.split(".");
  return parts.length > 1 ? parts.pop()!.toLowerCase() : "";
}

export function isSupportedExtension(extension: string): boolean {
  return SUPPORTED.has(extension.toLowerCase());
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
