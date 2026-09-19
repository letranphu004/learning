# AI Engineering từ A-Z: Học qua dự án RAG Study Assistant

> Tài liệu này tổng hợp toàn bộ kiến thức AI Engineering đã áp dụng khi xây
> dựng dự án `rag-study-assistant` trong repo này — từ khái niệm nền tảng
> (LLM, embedding, vector search) đến pipeline RAG hoàn chỉnh, kiến trúc hệ
> thống thực tế, và những lỗi thật đã gặp khi triển khai. Viết cho người mới
> bắt đầu: đi từ lý thuyết đơn giản đến ví dụ code cụ thể, không có phần nào
> chỉ là lý thuyết suông — mọi khái niệm đều trỏ về file thật trong
> `projects/rag-study-assistant/`.

**Ngày tổng hợp:** 2026-09-19

---

## Mục lục

1. [AI Engineering là gì?](#1-ai-engineering-là-gì)
2. [Nền tảng: LLM, Token, và Embedding](#2-nền-tảng-llm-token-và-embedding)
3. [Prompt Engineering](#3-prompt-engineering)
4. [RAG — Retrieval-Augmented Generation](#4-rag--retrieval-augmented-generation)
5. [Kiến trúc hệ thống thực tế](#5-kiến-trúc-hệ-thống-thực-tế)
6. [Observability & Evaluation](#6-observability--evaluation)
7. [Framework vs tự viết tay: LangChain](#7-framework-vs-tự-viết-tay-langchain)
8. [Bài học thực chiến: lỗi thật đã gặp](#8-bài-học-thực-chiến-lỗi-thật-đã-gặp)
9. [Thuật ngữ & lộ trình học tiếp theo](#9-thuật-ngữ--lộ-trình-học-tiếp-theo)

---

## 1. AI Engineering là gì?

**AI Engineering** là công việc *xây dựng sản phẩm dùng được* trên nền một mô
hình AI có sẵn (LLM, embedding model...) — khác với:

- **AI/ML Research**: tạo ra kiến trúc mô hình mới, cải tiến thuật toán huấn
  luyện. Bạn không làm việc này khi dùng Ollama chạy `llama3.2:3b` — bạn dùng
  một mô hình người khác đã huấn luyện sẵn.
- **Data Science**: phân tích dữ liệu để rút ra insight, thường không cần
  triển khai hệ thống chạy 24/7 phục vụ người dùng thật.
- **MLOps**: vận hành, giám sát, và scale các hệ thống ML ở quy mô lớn
  (nhiều server, nhiều mô hình, CI/CD cho model).

AI Engineer đứng giữa: nhận một mô hình đã có (qua API như OpenAI, hoặc chạy
local như Ollama), rồi **thiết kế pipeline** biến nó thành sản phẩm thật —
ingest dữ liệu, retrieval, prompt, xử lý lỗi, đo lường chất lượng, giao diện
người dùng. Đây chính xác là những gì dự án `rag-study-assistant` trong repo
này đã làm qua 7 phase.

**Lộ trình học của tài liệu này** đi theo đúng thứ tự dự án đã triển khai:
nền tảng lý thuyết (mục 2-3) → kỹ thuật RAG (mục 4) → cách ráp thành hệ thống
chạy được (mục 5) → cách biết hệ thống có tốt không (mục 6) → so sánh cách
làm thủ công với dùng framework (mục 7) → những cái bẫy thực tế đã rơi vào
(mục 8).

---

## 2. Nền tảng: LLM, Token, và Embedding

### LLM (Large Language Model) hoạt động thế nào — hiểu ở mức đủ dùng

Một LLM (như `llama3.2:3b` trong dự án) about bản chất là một hàm toán học
khổng lồ: nhận vào một chuỗi văn bản, dự đoán **từ tiếp theo có khả năng
cao nhất**, lặp lại quá trình đó cho đến khi đủ câu trả lời. Nó không "biết"
sự thật — nó chỉ giỏi dự đoán từ nào hợp lý tiếp theo dựa trên dữ liệu đã
học. Đây là lý do tại sao nó có thể "bịa" (hallucinate) thông tin nghe rất
tự nhiên nhưng sai — và tại sao kỹ thuật RAG (mục 4) lại quan trọng: nó ép
mô hình trả lời dựa trên tài liệu thật thay vì chỉ dựa vào trí nhớ huấn
luyện.

### Token — đơn vị mà LLM thực sự "nhìn thấy"

LLM không đọc từng ký tự hay từng từ — nó đọc **token**, là các mảnh văn bản
được cắt theo một bộ từ điển cố định (ví dụ từ "chunking" có thể bị cắt
thành "chunk" + "ing"). Điều này quan trọng thực tế vì:

- Model có **giới hạn context window** (số token tối đa xử lý một lần) — đây
  chính là lý do phải chunking tài liệu dài (mục 4).
  ```12:15:projects/rag-study-assistant/backend/app/ingestion.py
  """Turn a raw document into plain text, then into overlapping chunks.

  Chunking here is character-based, not token-based. That's a deliberate
  simplification for a learning project: token-accurate chunking needs a
  ```
  Dự án này **cố tình đơn giản hóa**: đếm ký tự (`chunk_size=800`) thay vì
  đếm token thật, vì đếm token chính xác cần thêm một tokenizer khớp với
  model embedding — một điểm phức tạp nữa để học sau, không phải bây giờ.
- `response_chars` trong log observability (mục 6) cũng cố tình đặt tên
  "chars" chứ không phải "tokens" — vì đó chỉ là số ký tự, một phép xấp xỉ
  thô, không phải số token thật. Đặt tên trung thực với những gì đo được là
  một nguyên tắc kỹ thuật quan trọng.

### Embedding — biến văn bản thành con số để so sánh ý nghĩa

**Embedding** là một vector số (ví dụ 768 chiều) đại diện cho *ý nghĩa* của
một đoạn văn bản. Hai câu có ý nghĩa gần nhau sẽ có embedding vector "gần
nhau" trong không gian nhiều chiều đó (đo bằng khoảng cách cosine hoặc
Euclidean). Đây là nền tảng của **semantic search** — tìm kiếm theo ý nghĩa
thay vì khớp từ khóa chính xác.

Dự án dùng model `nomic-embed-text` (chạy qua Ollama) riêng biệt với model
chat `llama3.2:3b` — đây là điểm quan trọng: **embedding model và chat model
là hai mô hình khác nhau**, có vai trò khác nhau:

| | Chat model (`llama3.2:3b`) | Embedding model (`nomic-embed-text`) |
|---|---|---|
| Đầu vào | Chuỗi hội thoại | Một đoạn văn bản |
| Đầu ra | Văn bản trả lời | Vector số (không đọc được bằng mắt) |
| Dùng để | Sinh câu trả lời | Tìm kiếm/so sánh ý nghĩa |

```16:23:projects/rag-study-assistant/backend/app/llm_client.py
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in a single call via /api/embed.

        Ollama's /api/embed accepts a list under "input" and returns
        "embeddings" in the same order - no client-side batching needed.
        """
```

---

## 3. Prompt Engineering

### Cấu trúc hội thoại: system / user / assistant

Khi gọi một chat model, tin nhắn được cấu trúc thành danh sách các **role**:

- `system`: chỉ dẫn hành vi tổng thể của model (thường người dùng không thấy)
- `user`: câu hỏi/yêu cầu từ người dùng
- `assistant`: câu trả lời trước đó của model (dùng để duy trì ngữ cảnh hội
  thoại nhiều lượt)

Dự án xây prompt này trong `app/rag.py`:

```python
SYSTEM_PROMPT = (
    "You are a study assistant. Answer the question using ONLY the context "
    "below. If the context does not contain enough information to answer, "
    "say \"I don't know from the given documents\" instead of guessing.\n\n"
    "Context:\n{context}"
)

def build_prompt(question, chunks, history=None):
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        *(history or []),
        {"role": "user", "content": question},
    ]
```

### Grounding — ép model trả lời dựa trên sự thật, không bịa

**Grounding** (hay "faithfulness") là kỹ thuật ép model chỉ trả lời dựa trên
ngữ cảnh được cung cấp, thay vì dựa vào "trí nhớ" đã huấn luyện (thứ có thể
sai hoặc lỗi thời). Câu lệnh `"Answer using ONLY the context below... say
'I don't know'"` chính là grounding instruction.

**Bài học thực tế quan trọng** từ dự án: model nhỏ (`llama3.2:3b`, 3 tỷ
tham số) **không phải lúc nào cũng tuân theo instruction hoàn hảo** — đây là
giới hạn thật đã được ghi nhận trong phase spec của dự án (không phải lỗi
code, mà là giới hạn của bản thân model nhỏ), và là lý do tại sao **test với
model thật quan trọng hơn code review lý thuyết** — bạn phải tự kiểm chứng
model có thực sự "nghe lời" hay không.

### Prompt Engineering cơ bản cần nhớ

1. **Cụ thể, không mơ hồ**: "Trả lời dựa trên context" tốt hơn "Hãy hữu ích"
2. **Cho model một lối thoát trung thực**: dạy model nói "tôi không biết"
   thay vì đoán bừa khi thiếu thông tin — đây chính là điều `SYSTEM_PROMPT`
   ở trên làm.
3. **Không cần template engine phức tạp**: dự án cố tình giữ prompt là một
   Python string thuần (`SYSTEM_PROMPT.format(...)`) — không dùng Jinja2 hay
   framework prompt riêng, vì với 1-2 template đơn giản, một f-string là đủ
   (nguyên tắc YAGNI — "You Aren't Gonna Need It").

---

## 4. RAG — Retrieval-Augmented Generation

Đây là kỹ thuật cốt lõi của toàn bộ dự án. **RAG** kết hợp:

1. **Retrieval** (truy xuất): tìm những đoạn văn bản liên quan nhất trong
   kho tài liệu của bạn
2. **Generation** (sinh câu trả lời): đưa những đoạn đó vào prompt, để model
   trả lời dựa trên chúng

→ Model có thể trả lời chính xác về tài liệu **nó chưa từng được huấn
luyện**, mà không cần fine-tune lại model — đây là lý do RAG là kỹ thuật phổ
biến nhất để đưa "kiến thức riêng" vào LLM.

### Toàn bộ pipeline RAG, từng bước

```
Tài liệu (PDF/txt/md)
      │
      ▼
┌─────────────┐
│   INGEST    │  đọc file → trích text thô
└─────────────┘
      │
      ▼
┌─────────────┐
│   CHUNK     │  cắt văn bản dài thành đoạn nhỏ (có overlap)
└─────────────┘
      │
      ▼
┌─────────────┐
│   EMBED     │  mỗi chunk → 1 vector số (qua embedding model)
└─────────────┘
      │
      ▼
┌─────────────┐
│    STORE    │  lưu vector + text gốc vào Vector Database (Chroma)
└─────────────┘

  ... khi người dùng đặt câu hỏi ...

Câu hỏi người dùng
      │
      ▼
┌─────────────┐
│ EMBED câu   │  câu hỏi cũng được embed thành vector
│    hỏi      │
└─────────────┘
      │
      ▼
┌─────────────┐
│  RETRIEVE   │  tìm k chunk có vector "gần" vector câu hỏi nhất
└─────────────┘
      │
      ▼
┌─────────────┐
│  GENERATE   │  đưa k chunk + câu hỏi vào prompt → model sinh câu trả lời
└─────────────┘
```

### Bước 1-2: Ingest + Chunking

```39:58:projects/rag-study-assistant/backend/app/ingestion.py
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Fixed-size sliding-window chunking over characters.

    overlap must be < chunk_size or the window never advances.
    """
```

**Tại sao phải chunk (cắt nhỏ) tài liệu?**
- Embedding model và chat model đều có **giới hạn context window** (mục 2)
- Nếu nhét cả tài liệu 50 trang vào một lần, retrieval sẽ không thể tìm ra
  *đoạn* nào liên quan nhất — nó phải trả về cả tài liệu, làm loãng ngữ cảnh

**Tại sao cần overlap (chồng lấn) giữa các chunk?**
Nếu cắt cứng nhắc không chồng lấn, một câu quan trọng nằm ngay ranh giới
giữa 2 chunk có thể bị cắt đôi, làm mất ý nghĩa ở cả 2 phía. Overlap
(`overlap=100` ký tự) đảm bảo ngữ cảnh không bị "đứt gãy" ở ranh giới.

**Chiến lược chunking khác** (không dùng trong dự án này, nhưng nên biết):
- *Token-based chunking*: đếm token thật thay vì ký tự (chính xác hơn,
  phức tạp hơn — cần tokenizer khớp embedding model)
- *Semantic chunking*: cắt theo ranh giới ý nghĩa (đoạn văn, section) thay
  vì cắt cứng theo độ dài
- *Recursive chunking* (dùng trong LangChain, xem mục 7): thử cắt theo
  đoạn văn trước, nếu vẫn quá dài thì cắt theo câu, rồi mới cắt theo ký tự

### Bước 3-4: Embed + Store (Vector Database)

**Vector Database** (ở đây là **ChromaDB**) giải quyết bài toán: cho một
vector câu hỏi, tìm nhanh các vector "gần" nhất trong hàng nghìn/triệu vector
đã lưu — đây là bài toán **Approximate Nearest Neighbor (ANN) search**, quá
chậm nếu làm bằng vòng lặp so sánh tuyến tính khi dữ liệu lớn.

```19:35:projects/rag-study-assistant/backend/app/vectorstore.py
class ChromaStore:
    def __init__(self, persist_dir: str | None = None):
        self._client = chromadb.PersistentClient(path=str(persist_dir or settings.chroma_dir))
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME)

    def upsert(self, source: str, chunks: list[str], embeddings: list[list[float]]) -> int:
        """Replace all chunks for `source` with the given ones (idempotent
        re-ingestion: delete-then-add rather than a real upsert, since chunk
        boundaries can shift between re-ingests of an edited document)."""
```

**Điểm thiết kế quan trọng: idempotent re-ingest.** Nếu bạn ingest lại cùng
một tài liệu (ví dụ sau khi sửa nội dung), hệ thống **xóa hết chunk cũ của
nguồn đó rồi thêm chunk mới** (`delete` rồi `add`), thay vì cố "update" từng
chunk — vì ranh giới chunk có thể đã thay đổi hoàn toàn sau khi sửa tài
liệu. Đây là một pattern quan trọng cần nhớ khi thiết kế bất kỳ pipeline
ingest nào.

**Chroma chạy "embedded"** — nghĩa là nó chạy ngay trong tiến trình Python
của bạn, lưu dữ liệu vào một thư mục trên đĩa (`backend/data/chroma/`),
**không cần** một server database riêng biệt như PostgreSQL hay MongoDB.
Đây là lựa chọn tốt cho dự án học tập/local-first, nhưng ở quy mô production
lớn (nhiều server, dữ liệu khổng lồ), người ta thường dùng vector database
dạng service riêng (Pinecone, Qdrant, Weaviate, hoặc pgvector trên
PostgreSQL).

### Bước 5-6: Retrieve + Generate

```30:39:projects/rag-study-assistant/backend/app/rag.py
async def retrieve_chunks(question: str, store: ChromaStore, llm: OllamaClient) -> list[dict]:
    [embedding] = await llm.embed([question])
    return store.query(embedding, k=settings.retrieval_k)
```

`k` (mặc định = 4) là **số chunk lấy về** cho mỗi câu hỏi. Đây là một tham số
đánh đổi quan trọng:
- `k` quá nhỏ → có thể bỏ sót thông tin liên quan
- `k` quá lớn → prompt bị loãng với nội dung không liên quan, tốn nhiều
  token hơn, chậm hơn

Không có con số "đúng" tuyệt đối — đây chính xác là điều mục 6
(Evaluation) giúp bạn đo lường và tinh chỉnh dựa trên dữ liệu thật, không
phải đoán mò.

---

## 5. Kiến trúc hệ thống thực tế

Dự án `rag-study-assistant` là một ví dụ đầy đủ về cách ráp các khái niệm
trên thành một **hệ thống chạy được, có giao diện, có state**:

```
┌──────────────────┐         HTTP/SSE         ┌───────────────────────┐
│    Frontend       │ ──────────────────────► │       Backend          │
│  (HTML/CSS/JS      │                         │   (FastAPI, Python)    │
│   thuần, không có   │ ◄────────────────────── │                         │
│   build tool)        │      JSON/stream        │  ┌──────────────────┐  │
└──────────────────┘                          │  │  ChromaDB (vector) │  │
     port 8080                                 │  │  SQLite (memory,   │  │
                                                │  │   observability)   │  │
                                                │  └──────────────────┘  │
                                                │           │             │
                                                │           ▼             │
                                                │  ┌──────────────────┐  │
                                                │  │   Ollama (LLM)    │  │
                                                │  │  chạy local, port  │  │
                                                │  │      11434         │  │
                                                │  └──────────────────┘  │
                                                └───────────────────────┘
                                                        port 8000
```

### Vì sao chạy 100% local (Ollama thay vì OpenAI API)?

- **Không tốn phí API**, không cần API key
- **Riêng tư dữ liệu**: tài liệu của bạn không rời khỏi máy
- **Học được cơ chế thật**: khi tự quản lý model, bạn hiểu rõ latency, giới
  hạn context, cách stream token hoạt động ở tầng thấp — điều bị framework/
  API ẩn đi

Đánh đổi: model local nhỏ hơn (3B tham số) yếu hơn GPT-4/Claude, chạy chậm
hơn trên máy không có GPU mạnh.

### Streaming response (Server-Sent Events)

Thay vì chờ model sinh **toàn bộ** câu trả lời rồi mới hiển thị (cảm giác
"đứng hình" vài giây), streaming hiển thị **từng token ngay khi model sinh
ra nó** — giống ChatGPT/Claude hiển thị chữ chạy dần.

```45:63:projects/rag-study-assistant/backend/app/llm_client.py
    async def chat_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Streaming chat completion. Ollama's streaming response is
        newline-delimited JSON (one object per line), not SSE - we
        translate each line's content delta into plain text chunks and let
        the caller (the /chat/stream route) re-wrap them as SSE."""
```

**Điểm kỹ thuật quan trọng:** Ollama trả về stream dạng **NDJSON** (mỗi
dòng một JSON object), còn giao thức chuẩn giữa backend-frontend là **SSE**
(Server-Sent Events, định dạng `data: {...}\n\n`). Backend phải **dịch**
giữa hai định dạng này — một ví dụ thực tế về việc các hệ thống khác nhau
hiếm khi dùng chung một giao thức, và việc "dịch" giữa chúng là công việc
thường xuyên của backend engineer.

Ở phía frontend, vì `EventSource` (API chuẩn cho SSE) **không hỗ trợ gửi
POST body**, dự án dùng `fetch()` + đọc `response.body.getReader()` thủ
công để vẫn nhận được luồng SSE trong khi gửi được JSON body (`session_id`,
`message`) — một giới hạn thực tế của web platform mà bạn sẽ gặp lại nếu tự
xây tính năng streaming chat.

### Conversation Memory — hội thoại nhiều lượt

Để hỏi tiếp "vậy còn cái đó thì sao?" và model hiểu "cái đó" là gì, hệ thống
cần lưu **lịch sử hội thoại** và gửi lại vào prompt mỗi lượt hỏi mới.

```29:39:projects/rag-study-assistant/backend/app/memory.py
    def recent(self, session_id: str, n: int = 6) -> list[dict]:
        """Last n messages for a session, oldest first (ready to prepend to
        a prompt's messages list)."""
```

Hai quyết định thiết kế quan trọng:
1. **`session_id`** được frontend tự sinh (`crypto.randomUUID()`) và lưu
   trong `localStorage` của trình duyệt — nghĩa là mỗi trình duyệt/thiết bị
   có một cuộc hội thoại riêng, và refresh trang vẫn giữ được ngữ cảnh (vì
   `localStorage` tồn tại qua các lần tải trang).
2. **Giới hạn số lượt lịch sử** (`n=6`, tức 3 lượt hỏi-đáp gần nhất) — nếu
   không giới hạn, prompt sẽ **phình to vô hạn** theo thời gian, vừa tốn
   token vừa có thể vượt giới hạn context window của model. Lịch sử cũ vẫn
   được lưu trong database, chỉ là không gửi vào prompt nữa.

---

## 6. Observability & Evaluation

Đây là điều phân biệt **"một demo chạy được"** với **"một hệ thống AI
Engineer thực thụ xây dựng"**: khả năng *đo lường* xem hệ thống có đang hoạt
động tốt hay không, thay vì chỉ tin vào cảm giác "nhìn có vẻ ổn".

### Observability — ghi log mọi request

```1:27:projects/rag-study-assistant/backend/app/observability.py
"""Best-effort structured logging of every RAG request, same stdlib
sqlite3-no-ORM pattern as memory.py - shares the one `app.db` file (a
separate `interactions` table) rather than adding a second database file."""
```

Mỗi lần gọi `/chat`, hệ thống ghi lại: câu hỏi, các chunk đã truy xuất
(kèm "khoảng cách" tới câu hỏi), câu trả lời, **độ trễ (latency)**, và số
ký tự của prompt/response. Đây chính là dữ liệu bạn cần để trả lời các câu
hỏi thực tế: "Tại sao câu trả lời này chậm?", "Model có đang truy xuất đúng
chunk không?", "Có câu hỏi nào bị trả lời sai lặp lại nhiều lần không?"

**Nguyên tắc thiết kế quan trọng: logging không được làm hỏng tính năng
chính.** Nếu việc ghi log bị lỗi (database khóa, ổ đĩa đầy...), hệ thống vẫn
phải trả lời người dùng bình thường — chỉ in lỗi ra `stderr` để debug sau.
Đây là nguyên tắc **"best-effort, never fail the main path"** áp dụng cho
mọi hệ thống logging/monitoring trong thực tế production.

### Evaluation — đo lường chất lượng retrieval bằng con số

**Câu hỏi cốt lõi:** làm sao biết hệ thống RAG của bạn "tốt"? Cảm giác chủ
quan ("tôi thấy câu trả lời ổn") không đủ tin cậy và không đo lường được
theo thời gian khi bạn thay đổi `chunk_size`, `k`, hay embedding model.

**`precision@k`** là một metric chuẩn trong Information Retrieval:

> Trong số **k** chunk được truy xuất, bao nhiêu phần trăm thực sự đúng
> nguồn tài liệu mong đợi?

```python
matches = sum(1 for r in results if r["source"] == expected_source)
precision = matches / len(results) if results else 0.0
```

Bạn cần một **bộ dữ liệu đánh giá** (`eval_dataset.json`) — một danh sách
câu hỏi kèm "đáp án đúng nên đến từ nguồn nào" được **gán nhãn thủ công**
bởi con người. Sau đó chạy script đo tự động mỗi khi thay đổi cấu hình, để
biết thay đổi đó làm hệ thống **tốt hơn hay tệ đi** — thay vì đoán.

**Bài học thực tế quan trọng đã gặp trong dự án này:** công thức tính
`precision@k` ban đầu chia cho **k yêu cầu** thay vì **số chunk thực sự trả
về** — khi kho tài liệu nhỏ hơn `k`, ChromaDB tự động giới hạn số kết quả
trả về, khiến công thức sai này làm điểm số **giống hệt nhau ở mọi câu hỏi**
(không phân biệt được câu hỏi tốt/xấu) — một bug thật được tìm ra qua code
review, minh chứng rằng **ngay cả công thức đo lường cũng cần được kiểm
chứng bằng dữ liệu thật**, không chỉ tin vào lý thuyết sách vở.

---

## 7. Framework vs tự viết tay: LangChain

Dự án xây **hai backend song song** để so sánh trực tiếp: một tự viết tay
(`backend/`), một dùng framework LangChain (`backend_langchain/`) — cùng
model, cùng tham số, khác nhau duy nhất ở việc có dùng framework hay không.

### LangChain giải quyết vấn đề gì?

LangChain là framework phổ biến nhất để xây ứng dụng LLM, cung cấp sẵn:
- **Document loaders**: đọc PDF/txt/... thành `Document` object
  (`PyPDFLoader`, `TextLoader`)
- **Text splitters**: chunking sẵn (`RecursiveCharacterTextSplitter`)
- **Vector store wrappers**: giao diện thống nhất cho nhiều vector DB khác
  nhau (Chroma, Pinecone, Qdrant...) — đổi database không cần viết lại code
- **LCEL (LangChain Expression Language)**: ghép các bước thành pipeline
  bằng cú pháp `|` (pipe), ví dụ: `retriever | prompt | llm | parser`

### Những gì thực sự quan sát được (không phải lý thuyết suông)

Từ so sánh thật trong `docs/langchain-comparison.md` của dự án:

1. **Tổng số dòng code không giảm nhiều như kỳ vọng** — LangChain tiết kiệm
   phần *retrieval internals* (chunking, vector store wrapper), nhưng phần
   "glue code" (FastAPI route, xử lý request/response) vẫn tương đương.
2. **Lấy cả câu trả lời VÀ nguồn trích dẫn từ một LCEL chain duy nhất khá
   rắc rối** — chain một-đường-ống tự nhiên (`retriever | format | prompt |
   llm | parser`) làm mất luôn danh sách `Document` đã truy xuất trước khi
   nó tới response. Cần thêm kỹ thuật (`RunnableParallel`) để lấy cả hai —
   trong bản tự viết tay, việc này *miễn phí* (biến `chunks` vốn đã có sẵn
   trong scope).
3. **`PyPDFLoader` chỉ đọc từ đường dẫn file, không đọc bytes trực tiếp** —
   một PDF upload qua HTTP phải ghi ra file tạm trước, trong khi bản tự viết
   tay dùng `pypdf.PdfReader(io.BytesIO(...))` đọc thẳng từ bytes.
4. **Sự "trôi dạt" của dependency là có thật**: khi cài `langchain-community`
   (cần cho `PyPDFLoader`), pip in ra cảnh báo package này **đang bị ngừng
   bảo trì**, khuyến nghị chuyển sang package độc lập khác — một ví dụ thật
   về việc hệ sinh thái framework AI thay đổi rất nhanh.

### Khi nào nên dùng framework, khi nào nên tự viết tay?

| Tình huống | Nên chọn |
|---|---|
| Học để hiểu cơ chế RAG hoạt động thế nào | Tự viết tay (như phase 1-6 của dự án) |
| Cần đổi qua lại nhiều vector DB/LLM provider | Framework (interface thống nhất) |
| Pipeline đơn giản, ít thay đổi | Tự viết tay (ít dependency, dễ debug) |
| Cần các pattern phức tạp có sẵn (agent, tool-calling, multi-step reasoning) | Framework (đỡ phải tự xây từ đầu) |
| Production cần kiểm soát chặt latency/chi phí | Tự viết tay hoặc framework nhưng đo lường kỹ (mục 6) |

---

## 8. Bài học thực chiến: lỗi thật đã gặp

Đây là phần giá trị nhất để học — không phải lý thuyết, mà là **lỗi thật đã
xảy ra trong chính dự án này**, cách phát hiện, và cách sửa.

### Lỗi 1: Nuốt lỗi im lặng (Silent Error Swallowing)

**Chuyện gì xảy ra:** `OllamaClient.chat_stream` chỉ đọc field `message.content`
từ mỗi dòng NDJSON, không kiểm tra field `error`. Nếu Ollama trả lỗi giữa
chừng (ví dụ cấu hình sai tên model), lỗi đó bị "nuốt" thành chuỗi rỗng —
người dùng nhận được câu trả lời trống, không có thông báo lỗi nào cả, và
hệ thống còn lưu luôn "câu trả lời rỗng" đó vào lịch sử hội thoại.

**Cách phát hiện:** code review có chủ đích tìm câu hỏi "nếu hàm này gặp
input bất thường thì sao?" thay vì chỉ đọc code xem nó làm gì khi mọi thứ
suôn sẻ.

**Bài học:** Khi tích hợp với một API bên ngoài (kể cả chạy local như
Ollama), **đừng giả định** lỗi luôn thể hiện qua HTTP status code. Một số
API (như Ollama streaming) nhúng lỗi *bên trong* body của response 200 OK —
bạn phải chủ động parse và kiểm tra.

### Lỗi 2: Rò rỉ dữ liệu test vào database thật (Test Isolation)

**Chuyện gì xảy ra:** Sau khi thêm tính năng observability logging, các
test tự động gọi `/chat` để kiểm tra tính năng chat — nhưng quên "thay thế"
(override) đối tượng logger bằng một bản dùng database tạm, giống như đã
làm với `store` (Chroma) và `memory` (lịch sử hội thoại). Kết quả: mỗi lần
chạy test suite, các câu hỏi/câu trả lời của test bị ghi thẳng vào
`app.db` **thật** trên máy dev.

**Cách phát hiện:** phát hiện thủ công khi query database thật và thấy có
câu hỏi lạ như "What is the capital of France?" (một câu test) lẫn trong dữ
liệu thật.

**Bài học:** Khi một module mới (observability) chia sẻ **tài nguyên có
trạng thái** (ở đây là cùng file database) với các module cũ đã có test,
**mọi** test chạm vào tài nguyên đó phải được audit lại để đảm bảo cô lập
(isolation) — không chỉ module mới viết ra mà cả các test cũ hiện đang gọi
vào đường code đã bị thay đổi.

### Lỗi 3: CORS bị chặn vì mở sai kiểu URL

**Chuyện gì xảy ra:** Người dùng gặp lỗi `Failed to fetch` khi thử chat qua
giao diện web. Nguyên nhân: mở file `index.html` trực tiếp bằng cách
double-click (`file:///...`) thay vì qua HTTP server
(`http://127.0.0.1:8080/...`). Khi mở theo kiểu `file://`, trình duyệt gửi
`Origin: null`, không nằm trong danh sách CORS được backend cho phép → bị
chặn.

**Bài học:** CORS (Cross-Origin Resource Sharing) là cơ chế bảo mật của
trình duyệt kiểm tra **origin** (giao thức + domain + port) của trang gọi
API, không phải chỉ domain. `http://127.0.0.1:8080` và
`file:///path/to/file.html` là hai origin hoàn toàn khác nhau dưới góc nhìn
CORS, dù file HTML giống hệt nhau.

### Lỗi 4: Công thức đo lường sai (đã nói ở mục 6)

Đã phân tích chi tiết ở mục 6 — nhắc lại ngắn gọn: **ngay cả code đo lường
chất lượng cũng có thể có bug**, và cách duy nhất để phát hiện là chạy nó
với dữ liệu thật và kiểm tra xem kết quả có *ý nghĩa* hay không (ở đây: tất
cả câu hỏi ra cùng một điểm số là dấu hiệu đáng ngờ, không phải điều bình
thường).

### Điểm chung của cả 4 bài học

Không có lỗi nào ở trên bị phát hiện chỉ bằng cách "đọc code thấy có vẻ
đúng". Cả 4 đều được tìm ra bằng cách **chạy hệ thống thật** (real Ollama,
real database, real browser) và **quan sát hành vi thực tế**, thường qua sự
kết hợp của: test tự động + code review có chủ đích tìm edge case + gỡ lỗi
trực tiếp khi người dùng báo lỗi. Đây là kỹ năng quan trọng nhất của một AI
Engineer: **không tin vào code cho đến khi thấy nó chạy đúng với dữ liệu và
điều kiện thật**.

---

## 9. Thuật ngữ & lộ trình học tiếp theo

### Glossary — thuật ngữ cần nhớ

| Thuật ngữ | Giải thích ngắn gọn |
|---|---|
| **LLM** | Large Language Model — mô hình ngôn ngữ lớn, dự đoán từ tiếp theo |
| **Token** | Đơn vị văn bản nhỏ nhất mà LLM xử lý (không phải từ, không phải ký tự) |
| **Embedding** | Vector số đại diện cho ý nghĩa của văn bản |
| **Vector Database** | Database chuyên tìm kiếm các vector "gần" nhau về ý nghĩa |
| **RAG** | Retrieval-Augmented Generation — kết hợp truy xuất tài liệu + sinh câu trả lời |
| **Chunking** | Cắt tài liệu dài thành đoạn nhỏ trước khi embed |
| **Grounding** | Ép model trả lời dựa trên dữ liệu cho sẵn, không bịa |
| **Hallucination** | Model "bịa" thông tin nghe hợp lý nhưng sai sự thật |
| **Context window** | Giới hạn số token model xử lý được trong một lần gọi |
| **Streaming (SSE)** | Trả kết quả dần dần (token-by-token) thay vì chờ xong hết |
| **Precision@k** | Metric đo: trong k kết quả truy xuất, bao nhiêu % đúng |
| **LCEL** | LangChain Expression Language — cú pháp ghép pipeline bằng `\|` |
| **Idempotent** | Làm lại nhiều lần cho kết quả giống hệt (không tạo trùng lặp) |
| **Observability** | Khả năng quan sát/đo lường hành vi hệ thống qua log, metric |

### Lộ trình học tiếp theo (gợi ý, theo độ khó tăng dần)

1. **Cải thiện chunking**: thử semantic chunking hoặc token-based chunking
   thay vì character-based, đo bằng `eval_retrieval.py` xem có cải thiện
   `precision@k` không.
2. **Reranking**: sau khi retrieval trả về top-k thô, dùng một model nhỏ
   hơn (cross-encoder) để **xếp hạng lại** — kỹ thuật phổ biến để tăng chất
   lượng RAG mà không cần embedding model tốt hơn.
3. **Agents & Tool-calling**: cho LLM khả năng *tự quyết định* gọi công cụ
   nào (tìm kiếm web, tính toán, gọi API khác) thay vì chỉ trả lời từ
   context tĩnh — bước tiếp theo tự nhiên sau RAG.
4. **Multi-modal RAG**: mở rộng ngoài văn bản — retrieval trên hình ảnh,
   bảng biểu trong PDF (không chỉ text thô như `pypdf` hiện tại).
5. **Fine-tuning**: khi nào RAG không đủ (cần thay đổi *cách* model trả
   lời, không chỉ *nội dung* nó biết) thì bắt đầu học fine-tuning.
6. **Guardrails & Safety**: lọc output độc hại, ngăn prompt injection từ
   tài liệu được ingest (một tài liệu độc hại có thể chứa chỉ dẫn giả mạo
   nhắm vào chính hệ thống RAG — rủi ro bảo mật thực tế cần biết).
7. **MLOps cho LLM**: CI/CD cho prompt, A/B test giữa các phiên bản prompt/
   model, giám sát chi phí và latency ở quy mô nhiều người dùng đồng thời.

### Cách dùng tài liệu này tiếp theo

Tài liệu này được lưu tại `docs/ai-engineering-knowledge-base.md` trong
repo — mọi ví dụ code đều trỏ về file thật trong
`projects/rag-study-assistant/`. Khi thêm phase mới (phase 6 docker, hoặc
mở rộng thêm sau này), quay lại cập nhật thêm mục tương ứng để tài liệu này
luôn phản ánh đúng những gì đã thực sự xây dựng — không phải lý thuyết
tách rời khỏi code.
