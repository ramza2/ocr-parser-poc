import { useRef, useState } from "react";
import type { BoundingBox } from "../../types/vlm";

interface BboxItem {
  text: string;
  bbox?: BoundingBox | null;
  confidence?: number | null;
}

interface Props {
  imageUrl: string;
  items: BboxItem[];
  highlightIndex: number | null;
}

function bboxColor(confidence: number | null | undefined, highlight: boolean) {
  if (highlight) return "rgba(59, 130, 246, 0.7)";
  if (confidence == null) return "rgba(100, 116, 139, 0.4)";
  if (confidence >= 0.9) return "rgba(16, 185, 129, 0.5)";
  if (confidence >= 0.7) return "rgba(245, 158, 11, 0.5)";
  return "rgba(239, 68, 68, 0.5)";
}

export default function ImageWithBbox({ imageUrl, items, highlightIndex }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });

  const handleLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
  };

  return (
    <div ref={containerRef} className="relative inline-block max-w-full">
      <img
        src={imageUrl}
        alt="업로드 이미지"
        onLoad={handleLoad}
        className="max-h-[500px] max-w-full rounded-lg"
      />
      {imgSize.w > 0 && (
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full"
          viewBox={`0 0 ${imgSize.w} ${imgSize.h}`}
          preserveAspectRatio="xMidYMid meet"
        >
          {items.map((item, i) => {
            if (!item.bbox) return null;
            const { x, y, width, height } = item.bbox;
            const highlight = i === highlightIndex;
            return (
              <rect
                key={i}
                x={x * imgSize.w}
                y={y * imgSize.h}
                width={width * imgSize.w}
                height={height * imgSize.h}
                fill="none"
                stroke={bboxColor(item.confidence, highlight)}
                strokeWidth={highlight ? 3 : 2}
              />
            );
          })}
        </svg>
      )}
    </div>
  );
}
