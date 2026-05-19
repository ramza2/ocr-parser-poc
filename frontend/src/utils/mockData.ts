import type { ParseResponse } from "../types/parser";

export const MOCK_PARSE_RESULT: ParseResponse = {
  success: true,
  parser_id: "PDF_EASYOCR",
  file_name: "sample_invoice_2023.pdf",
  extension: "pdf",
  elapsed_ms: 1240,
  page_count: 3,
  text_length: 4281,
  table_count: 0,
  error_count: 0,
  text: `[페이지 1]
INVOICE
Invoice No: INV-2023-001
Date: 2023-10-24

Bill To:
TechSolutions Inc.
123 Innovation Dr, Silicon Valley, CA

[페이지 2]
Line Items:
1. Enterprise Software License
2. Premium Support Package

[페이지 3]
Total Amount Due: $12,450.00
Thank you for your business.`,
  pages: [
    {
      page_no: 1,
      text: "INVOICE\nInvoice No: INV-2023-001\nDate: 2023-10-24",
      blocks: [],
    },
    {
      page_no: 2,
      text: "Line Items:\n1. Enterprise Software License",
      blocks: [],
    },
    { page_no: 3, text: "Total Amount Due: $12,450.00", blocks: [] },
  ],
  tables: [],
  logs: [
    { level: "INFO", message: "파일 업로드 완료", timestamp: "10:00:01" },
    { level: "INFO", message: "스캔 PDF 페이지 OCR (easyocr)", timestamp: "10:00:02" },
    { level: "INFO", message: "스캔 PDF 3페이지 렌더링 완료", timestamp: "10:00:02" },
    { level: "INFO", message: "페이지 1 OCR 중...", timestamp: "10:00:02" },
    { level: "INFO", message: "페이지 2 OCR 중...", timestamp: "10:00:03" },
    { level: "INFO", message: "페이지 3 OCR 중...", timestamp: "10:00:03" },
    { level: "INFO", message: "스캔 PDF OCR 완료", timestamp: "10:00:04" },
  ],
  errors: [],
};
