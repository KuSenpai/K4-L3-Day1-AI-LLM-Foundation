# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Ở cả 4 mức temperature, model đều cố gắng tìm một sự thật thú vị về Việt Nam nhưng cách lựa chọn và diễn đạt thay đổi. Temperature càng cao thì nội dung càng đa dạng nhưng cũng khó đoán hơn. Ở Temp 0.0 thì câu trả lời thận trọng và tập trung vào ngôn ngữ, còn ở temp 0.5 đến 1.5 thì câu trả lời có phần bay bổng và nhiều hướng khác nhau.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Tôi sẽ đặt temperature khoảng 0.2–0.4 cho chatbot hỗ trợ khách hàng. Mức này giúp câu trả lời ổn định, nhất quán và phù hợp với các câu hỏi thường gặp, đồng thời vẫn đủ tự nhiên khi giao tiếp. Với các thông tin liên quan đến chính sách, đơn hàng hoặc thanh toán, tính chính xác và nhất quán quan trọng hơn sự sáng tạo.*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> *Theo bảng giá, GPT-4o có giá output khoảng 0.010 USD/1K token, còn GPT-4o-mini có giá khoảng 0.0006 USD/1K token. Vì vậy, GPT-4o đắt hơn khoảng: 0.010 / 0.0006 ≈ 16.7 lần*
*Với 10.000 người dùng, mỗi người gọi 3 lần và mỗi lần sinh 350 token, tổng output là khoảng 10,5 triệu token mỗi ngày. GPT-4o phù hợp với các yêu cầu phức tạp cần chất lượng suy luận và độ chính xác cao, chẳng hạn phân tích tài liệu pháp lý hoặc hỗ trợ kỹ thuật chuyên sâu. GPT-4o-mini phù hợp với các tác vụ đơn giản, số lượng lớn như trả lời câu hỏi thường gặp, phân loại yêu cầu hoặc tạo phản hồi ngắn.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Độ dài: Phản hồi dành cho giáo viên tiểu học có xu hướng ngắn và tập trung vào ý tưởng cốt lõi: blockchain là một cuốn sổ dùng chung, mọi người cùng giữ bản sao và khó sửa dữ liệu. Phản hồi dành cho chuyên gia tài chính dài hơn vì cần giải thích thêm cấu trúc block, cách liên kết bằng hash, mạng lưới phi tập trung và cơ chế xác thực.*
*Từ vựng: Phản hồi cho trẻ em dùng từ quen thuộc như “sổ tay”, “bạn bè”, “trao đổi thẻ”, “viết bằng bút” và “chuỗi LEGO”. Phản hồi chuyên gia dùng thuật ngữ kỹ thuật như “distributed ledger”, “cryptographic hashing”, “consensus mechanism”, “Merkle tree”, “Proof of Work”, “Proof of Stake” và “smart contracts”.*
*Ví dụ: Ví dụ cho trẻ em dựa trên các đồ vật và hoạt động hằng ngày, giúp khái niệm trừu tượng dễ hình dung. Ví dụ cho chuyên gia dựa trên Bitcoin, Ethereum, giao dịch, node, SHA-256 và các cơ chế đồng thuận.*
*Ảnh hưởng của system prompt: System prompt đã định hướng model thay đổi vai trò, giọng điệu, độ sâu, vốn từ và cách chọn ví dụ theo đối tượng người đọc.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> *Theo công thức ước lượng của Part 1, đoạn văn có khoảng 133 token (100 / 0.75). Khi dùng count_tokens() với tiktoken, kết quả là khoảng 180 token, cao hơn khoảng 35% so với ước lượng. Sự chênh lệch xảy ra vì tokenizer không đếm token giống như đếm từ; một từ có thể được tách thành nhiều token tùy theo ký tự, dấu tiếng Việt và mức độ phổ biến của từ trong dữ liệu huấn luyện. Vì vậy, tiếng Việt đôi khi cần nhiều token hơn tiếng Anh có cùng số từ hoặc độ dài tương đương.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming quan trọng khi model cần tạo câu trả lời dài hoặc khi người dùng cần thấy phản hồi càng sớm càng tốt, chẳng hạn trong chatbot, trợ lý lập trình hoặc ứng dụng viết nội dung. Thay vì chờ toàn bộ câu trả lời hoàn thành, người dùng có thể bắt đầu đọc ngay khi những token đầu tiên xuất hiện, nên cảm giác chờ đợi ngắn hơn. Non-streaming phù hợp hơn với các tác vụ ngắn, các API backend cần nhận toàn bộ kết quả để xử lý tiếp, hoặc những trường hợp cần kiểm tra và lưu câu trả lời hoàn chỉnh trước khi hiển thị.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Exponential backoff giúp giảm dần tần suất retry khi API đang quá tải bằng cách tăng thời gian chờ sau mỗi lần thất bại, ví dụ 0.1, 0.2, 0.4 giây. So với delay cố định, cách này giảm áp lực lên server và tăng cơ hội request thành công khi hệ thống phục hồi. Nếu hàng nghìn client cùng retry với một delay cố định, chúng có thể gửi request lại cùng một thời điểm, tạo ra hiện tượng “thundering herd”, khiến server tiếp tục quá tải và các request lại đồng loạt thất bại.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *System promt: Bạn là một trợ giảng thân thiện của khóa học AI. Hãy trả lời ngắn gọn,rõ ràng bằng tiếng Việt, giải thích các khái niệm khó bằng ví dụ đơn giản và nói rõ khi bạn không chắc chắn về thông tin.*
*Chọn cụm “trợ giảng thân thiện” để định hướng giọng điệu dễ tiếp cận và phù hợp với người mới học. Cụm “ngắn gọn, rõ ràng bằng tiếng Việt” giúp câu trả lời không lan man và phù hợp với người dùng Việt Nam.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> *Hạn chế lớn nhất của trợ lý của tôi hiện tại là hiện tượng hallucination, tức là model có thể tự tạo ra thông tin không chính xác nhưng vẫn trình bày với vẻ chắc chắn. Điều này đặc biệt nguy hiểm khi người dùng hỏi về sự kiện, số liệu hoặc thông tin chuyên môn. Một cải thiện cụ thể là yêu cầu trợ lý nêu nguồn tham khảo cho các thông tin quan trọng, kết hợp với cơ chế RAG để tìm dữ liệu từ tài liệu đáng tin cậy trước khi trả lời. Nếu không tìm thấy thông tin phù hợp, trợ lý nên nói rõ rằng mình không chắc chắn thay vì tự đoán.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
