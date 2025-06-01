import os
import joblib
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel, TFRobertaModel
import logging
import warnings
import google.generativeai as genai
from config import API_KEY_GEMINI 
import pandas as pd
import gradio as gr

CSV_FILE_PATH = './chuong_trinh_dao_tao.csv'
MODEL_NAME = 'vinai/phobert-base'
MAX_LENGTH = 128
BASE_SAVE_DIR = './saved_model_output'
WEIGHTS_FILE_NAME = 'my_multitask_model_weights.weights.h5'
ENCODER_NGANH_FILE_NAME = 'encoder_nganh.joblib'
ENCODER_MUC_FILE_NAME = 'encoder_muc.joblib'

MODEL_WEIGHTS_LOAD_PATH = os.path.join(BASE_SAVE_DIR, WEIGHTS_FILE_NAME)
ENCODER_NGANH_PATH = os.path.join(BASE_SAVE_DIR, ENCODER_NGANH_FILE_NAME)
ENCODER_MUC_PATH = os.path.join(BASE_SAVE_DIR, ENCODER_MUC_FILE_NAME)

loaded_model = None
tokenizer_loaded = None
encoder_nganh_loaded = None
encoder_muc_loaded = None
num_classes_nganh_loaded = 0
num_classes_muc_loaded = 0
training_data_dict = {} 

def create_multitask_transformer_model(model_name, num_labels_nganh, num_labels_muc, max_length=MAX_LENGTH):
    base_transformer_layer = TFAutoModel.from_pretrained(model_name, from_pt=True, name="transformer_base_model")
    input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
    transformer_outputs = base_transformer_layer(input_ids, attention_mask=attention_mask)
    last_hidden_state = transformer_outputs.last_hidden_state
    cls_token_output = last_hidden_state[:, 0, :]
    dropout_layer = tf.keras.layers.Dropout(0.1)(cls_token_output)
    output_nganh = tf.keras.layers.Dense(num_labels_nganh, activation='softmax', name='output_nganh')(dropout_layer)
    output_muc = tf.keras.layers.Dense(num_labels_muc, activation='softmax', name='output_muc')(dropout_layer)
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=[output_nganh, output_muc])
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5) # Cần compile để load_weights không báo lỗi
    model.compile(optimizer=optimizer,
                  loss={'output_nganh': 'sparse_categorical_crossentropy',
                        'output_muc': 'sparse_categorical_crossentropy'},
                  metrics={'output_nganh': 'accuracy', 'output_muc': 'accuracy'})
    return model

def load_training_data_from_csv():
    global training_data_dict
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        for _, row in df.iterrows():
            key = (str(row['mã ngành']).strip(), str(row['mã mục']).strip()) 
            training_data_dict[key] = str(row['nội dung']).strip() 
        print(f"✅ Đã tải {len(training_data_dict)} mục từ dữ liệu đào tạo CSV.")
        return True
    except Exception as e:
        print(f"❌ Lỗi đọc file CSV '{CSV_FILE_PATH}': {e}")
        return False

def get_training_content(ma_nganh, ma_muc):
    return training_data_dict.get((ma_nganh, ma_muc), "Không tìm thấy nội dung tương ứng trong dữ liệu đào tạo đã tải.")

def load_all_dependencies():
    global loaded_model, tokenizer_loaded, encoder_nganh_loaded, encoder_muc_loaded
    global num_classes_nganh_loaded, num_classes_muc_loaded

    print("--- BẮT ĐẦU TẢI CÁC THÀNH PHẦN CẦN THIẾT ---")
    all_ok = True

    try:
        tokenizer_loaded = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("✅ Tokenizer đã tải.")
    except Exception as e:
        print(f"❌ Lỗi tải tokenizer: {e}")
        all_ok = False

    try:
        if os.path.exists(ENCODER_NGANH_PATH) and os.path.exists(ENCODER_MUC_PATH):
            encoder_nganh_loaded = joblib.load(ENCODER_NGANH_PATH)
            encoder_muc_loaded = joblib.load(ENCODER_MUC_PATH)
            num_classes_nganh_loaded = len(encoder_nganh_loaded.classes_)
            num_classes_muc_loaded = len(encoder_muc_loaded.classes_)
            print(f"✅ Đã tải encoders: Ngành ({num_classes_nganh_loaded} lớp) | Mục ({num_classes_muc_loaded} lớp)")
        else:
            print(f"❌ Lỗi: Không tìm thấy file encoder tại '{ENCODER_NGANH_PATH}' hoặc '{ENCODER_MUC_PATH}'.")
            all_ok = False
    except Exception as e:
        print(f"❌ Lỗi tải encoders: {e}")
        all_ok = False

    if all_ok and os.path.exists(MODEL_WEIGHTS_LOAD_PATH):
        try:
            if num_classes_nganh_loaded > 0 and num_classes_muc_loaded > 0:
                loaded_model = create_multitask_transformer_model(MODEL_NAME, num_classes_nganh_loaded, num_classes_muc_loaded)
                loaded_model.load_weights(MODEL_WEIGHTS_LOAD_PATH, by_name=True, skip_mismatch=True)
                print("✅ Trọng số mô hình đã tải.")
            else:
                print("❌ Lỗi: Thiếu thông tin số lớp từ encoders để tạo lại mô hình.")
                all_ok = False
        except Exception as e:
            print(f"❌ Lỗi khi tạo mô hình hoặc tải trọng số: {e}")
            all_ok = False
    elif all_ok and not os.path.exists(MODEL_WEIGHTS_LOAD_PATH):
        print(f"❌ Không tìm thấy file trọng số mô hình tại '{MODEL_WEIGHTS_LOAD_PATH}'.")
        all_ok = False
    
    if all_ok:
        print("✅ Tất cả các thành phần cần thiết đã được tải thành công!")
    else:
        print("❌ Một số thành phần không tải được, vui lòng kiểm tra lỗi.")
    return all_ok

def predict_classification(text):
    if not all([loaded_model, tokenizer_loaded, encoder_nganh_loaded, encoder_muc_loaded]):
        return "Lỗi: Mô hình chưa sẵn sàng.", "N/A", "N/A"
    try:
        encoding = tokenizer_loaded(text, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors='tf')
        inputs = {'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask']}
        pred_nganh_probs, pred_muc_probs = loaded_model.predict(inputs, verbose=0)
        
        pred_label_nganh_encoded = np.argmax(pred_nganh_probs, axis=1)[0]
        pred_label_muc_encoded = np.argmax(pred_muc_probs, axis=1)[0]
        
        ma_nganh_pred = encoder_nganh_loaded.inverse_transform([pred_label_nganh_encoded])[0]
        ma_muc_pred = encoder_muc_loaded.inverse_transform([pred_label_muc_encoded])[0]
        
        confidence_nganh = np.max(pred_nganh_probs)
        confidence_muc = np.max(pred_muc_probs)

        return f"📌 Dự đoán: Ngành = {ma_nganh_pred} (Độ tin cậy: {confidence_nganh:.2f}), Mục = {ma_muc_pred} (Độ tin cậy: {confidence_muc:.2f})", ma_nganh_pred, ma_muc_pred
    except Exception as e:
        return f"❌ Lỗi trong quá trình dự đoán phân loại: {e}", "N/A", "N/A"

def ask_gemini_with_context(user_question, ma_nganh, ma_muc):
    if ma_nganh == "N/A" or ma_muc == "N/A":
        return "Không thể lấy nội dung do lỗi phân loại trước đó."

    noi_dung_tham_khao = get_training_content(ma_nganh, ma_muc)
    
    try:
        genai.configure(api_key=API_KEY_GEMINI)
        generation_config = genai.types.GenerationConfig(temperature=0.7) # Điều chỉnh nhiệt độ nếu cần
        model_gemini = genai.GenerativeModel('gemini-2.0-flash', generation_config=generation_config) 
        
        prompt = f"""
Bạn là một trợ lý ảo thông minh, chuyên cung cấp thông tin về các chương trình đào tạo đại học.
Dựa trên câu hỏi của người dùng và thông tin tham khảo được cung cấp (nếu có), hãy đưa ra câu trả lời ngắn gọn, thu hẹp trong 1 đoạn từ 4 đến 6 câu, rõ ràng và hữu ích.

Thông tin tham khảo cho câu hỏi này:
- Ngành: {ma_nganh}
- Mục: {ma_muc}
- Nội dung chi tiết:
---
{noi_dung_tham_khao}
---

Câu hỏi của người dùng: "{user_question}"

Hãy trả lời câu hỏi trên. Nếu thông tin tham khảo không đủ hoặc không liên quan trực tiếp, hãy cố gắng trả lời dựa trên hiểu biết chung về cách các chương trình đào tạo thường được cấu trúc, nhưng nhấn mạnh rằng đó là thông tin chung. Nếu không thể trả lời, hãy nói rõ.
"""
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Lỗi khi gọi Gemini API: {e}"

def chatbot_response(user_question):
    classification_output, nganh_pred, muc_pred = predict_classification(user_question)
    
    gemini_answer = "Đang chờ phân loại câu hỏi..."
    if nganh_pred != "N/A" and muc_pred != "N/A":
        gemini_answer = ask_gemini_with_context(user_question, nganh_pred, muc_pred)
    elif "Lỗi" in classification_output : 
         gemini_answer = "Không thể xử lý do lỗi phân loại."


    return classification_output, gemini_answer

if __name__ == '__main__':
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings('ignore', category=UserWarning) 
    warnings.filterwarnings('ignore', category=FutureWarning) 

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Lỗi cấu hình GPU: {e}")

    if load_all_dependencies() and load_training_data_from_csv():
        print("🚀 Khởi tạo giao diện Gradio...")
        iface = gr.Interface(
            fn=chatbot_response,
            inputs=gr.Textbox(lines=3, placeholder="Nhập câu hỏi của bạn về chương trình đào tạo..."),
            outputs=[
                gr.Textbox(label="Kết quả Phân loại (Ngành/Mục)"),
                gr.Markdown(label="🤖 Trả lời từ Trợ lý AI")
            ],
            title="🎓 Trợ Lý Thông Tin Tuyển Sinh & Đào Tạo 🎓",
            description="Đặt câu hỏi về các ngành học và nhận câu trả lời được hỗ trợ bởi AI. Mô hình sẽ cố gắng phân loại câu hỏi của bạn vào ngành và mục phù hợp, sau đó sử dụng thông tin đó để tạo câu trả lời.",
            theme=gr.themes.Soft(),
            allow_flagging='never'
        )
        iface.launch(share=False)
    else:
        print("Không thể khởi chạy ứng dụng do lỗi tải các thành phần cần thiết.")