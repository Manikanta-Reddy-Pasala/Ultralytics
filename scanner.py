import os
import logging
import numpy as np
from ultralytics import YOLO
import ai_colormap
from datetime import datetime
import socket
import ai_model_pb2
from google.protobuf import message
import threading
from scanner_logging import setup_logging
import torch
import gc
import signal
import sys
import cv2


setup_logging()

###################PARAMETERS FROM THE HEADER##################################################
sample_rate = 30.72e6
center_freq = 938.9e6  # 4G
bandwidth = 58.6e6
num_center_frequencies = 3
overlap = 1
fft_size = 2048
dummy_file = "dummy.jpg"
num_khz_per_fft_point = 15
fifteen_mhz_points = int(15000 / num_khz_per_fft_point)
five_mhz_points = int(5000 / num_khz_per_fft_point)

edivide = 1e6
emul = 1e3
MAX_AI_BANDWIDTH = 60 * emul

# Singleton colormap objects - reuse across all requests
_normalizer = ai_colormap.NormalizePowerValue()
_color_mapper = ai_colormap.CustomImg()

# Thread concurrency limiter
_max_concurrent_threads = int(os.getenv('MAX_THREADS', '2'))
_thread_semaphore = threading.Semaphore(_max_concurrent_threads)


def sigterm_handler(_signo, _stack_frame):
    logging.info("Received SIGTERM, exiting...")
    sys.exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)


################################AI MODEL IMPORT#################################################
model_2g_new = YOLO("2G_MODEL/best_int8_openvino_model/", task="detect")
model_3g_4g_new = YOLO("3G_4G_MODEL/best.pt", task="detect")

###################################WARMUP FOR MODEL############################################
if os.path.exists(dummy_file):
    with torch.no_grad():
        logging.info("Warming up 2G model...")
        model_2g_new.predict(dummy_file, imgsz=[32, 32])
        logging.info("Warming up 3G/4G model...")
        model_3g_4g_new.predict(dummy_file, imgsz=[32, 32])
    gc.collect()
else:
    logging.warning(f"Warmup image '{dummy_file}' not found, skipping warmup. First inference will be slower.")

##################################CREATE SOCKET ##############################################
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def add_detected_cells_to_list(predicted_center_freq, clsitem, is_3g_4g, xval_list, xval, width,
                               freq_list_2g, freq_list_3g, freq_list_4g):
    if is_3g_4g:
        if clsitem == 0:
            xval_list.append([int(xval.item() - width.item() / 2), int(width.item())])
            if predicted_center_freq not in freq_list_3g:
                freq_list_3g.append(predicted_center_freq)
        elif clsitem == 1 or clsitem == 2:
            xval_list.append([int(xval.item() - width.item() / 2), int(width.item())])
            if predicted_center_freq not in freq_list_4g:
                freq_list_4g.append(predicted_center_freq)
    else:
        if predicted_center_freq not in freq_list_2g:
            freq_list_2g.append(predicted_center_freq)


def process_results_3g_4g(results_3g_4g, xval_list, start_freq, bandwidth,
                          freq_list_3g, freq_list_4g):
    for result in results_3g_4g:
        if len(result.boxes) != 0:
            pixel_size_of_img = result.orig_shape[1]
            boxes = result.boxes
            for idx, xywhn in enumerate(boxes.xywhn):
                clsitem = int(boxes.cls[idx].item())
                xval = boxes.xywh[idx][0]
                width = boxes.xywh[idx][2]
                predicted_center_freq = ((bandwidth / pixel_size_of_img) * xval)
                predicted_center_freq += start_freq
                predicted_center_freq /= edivide
                predicted_center_freq = round(float(predicted_center_freq), 1)
                add_detected_cells_to_list(predicted_center_freq,
                        clsitem, True, xval_list, xval, width,
                        None, freq_list_3g, freq_list_4g)
    xval_list.sort()
    return xval_list


def process_results_2g(results, start_freq_for_each_chunk, chunk_start_indexes_in_new_image, freq_list_2g):
    index_val = 0
    for result in results:
        if len(result.boxes) != 0:
            boxes = result.boxes
            for idx, xywhn in enumerate(boxes.xywhn):
                clsitem = int(boxes.cls[idx].item())
                xval = boxes.xywh[idx][0]
                x_val_in_image = xval.item()
                i = 0
                while i < len(chunk_start_indexes_in_new_image):
                    if x_val_in_image < chunk_start_indexes_in_new_image[i]:
                        index_val = i - 1
                    else:
                        index_val = i
                    i = i + 1

                predicted_center_freq = (15000 * (x_val_in_image - chunk_start_indexes_in_new_image[index_val]))
                start_freq_chunk_wise = start_freq_for_each_chunk[index_val]

                predicted_center_freq += start_freq_chunk_wise
                predicted_center_freq /= edivide
                predicted_center_freq = round(float(predicted_center_freq), 1)
                add_detected_cells_to_list(predicted_center_freq,
                        clsitem, False, None, 0, 0,
                        freq_list_2g, None, None)


def get_num_chunks_for_mem_optimization(bandwidth_recv, overlap):
    num_chunks = 1
    if bandwidth_recv >= 100 * emul:
        overlap = int(12 * emul / num_khz_per_fft_point)
        num_chunks = int(bandwidth_recv // (MAX_AI_BANDWIDTH - overlap * num_khz_per_fft_point))
    return num_chunks, overlap


def create_correct_spectrogram_by_rearranging_samples(num_center_frequencies, spectrogram, num_of_samples_in_freq):
    if num_center_frequencies == 1:
        return spectrogram

    parts = []
    if num_center_frequencies == 2:
        parts.append(spectrogram[0:num_of_samples_in_freq - 1, :fifteen_mhz_points])
        parts.append(spectrogram[num_of_samples_in_freq:num_of_samples_in_freq * 2 - 1, five_mhz_points:])
    else:
        loop_counter = 1
        while loop_counter <= num_center_frequencies - 1:
            if loop_counter == 1:
                parts.append(spectrogram[0:num_of_samples_in_freq - 1, :fifteen_mhz_points])
                parts.append(spectrogram[num_of_samples_in_freq:num_of_samples_in_freq * 2 - 1, five_mhz_points:fifteen_mhz_points])
            elif loop_counter == num_center_frequencies - 1:
                parts.append(spectrogram[num_of_samples_in_freq * loop_counter:num_of_samples_in_freq * (loop_counter + 1) - 1, five_mhz_points:])
            else:
                parts.append(spectrogram[num_of_samples_in_freq * loop_counter:num_of_samples_in_freq * (loop_counter + 1) - 1, five_mhz_points:fifteen_mhz_points])
            loop_counter += 1

    spectrogram_new = np.concatenate(parts, axis=1)
    del parts
    return spectrogram_new


def get_truncated_spectrum(spectrogram_new, num_chunks, chunk_iterator, center_freq_orig, num_samples_in_chunk, bandwidth, overlap):

    if num_chunks > 1:
        if chunk_iterator != 0:
            start_index = (chunk_iterator * num_samples_in_chunk) - (overlap * chunk_iterator)
            end_index = start_index + num_samples_in_chunk
            if end_index < spectrogram_new.shape[1]:
                spectrogram_predict = spectrogram_new[:, start_index:end_index]
                center_freq = center_freq_orig + int(MAX_AI_BANDWIDTH) * emul - overlap * num_khz_per_fft_point * emul
                bandwidth = MAX_AI_BANDWIDTH * emul
                center_freq_orig = center_freq
            else:
                end_index = spectrogram_new.shape[1]
                spectrogram_predict = spectrogram_new[:, start_index:end_index]
                bandwidth = (end_index - start_index) * num_khz_per_fft_point
                center_freq = center_freq_orig + int(MAX_AI_BANDWIDTH / 2) * emul + int(bandwidth / 2) * emul - overlap * num_khz_per_fft_point * emul
                bandwidth = bandwidth * emul
                center_freq_orig = center_freq
        else:
            spectrogram_predict = spectrogram_new[:, 0:num_samples_in_chunk]
            center_freq = (center_freq_orig - int((bandwidth / 2))) + (int((MAX_AI_BANDWIDTH / 2)) * emul)
            bandwidth = MAX_AI_BANDWIDTH * emul
            center_freq_orig = center_freq
    else:
        spectrogram_predict = None
        center_freq = None

    return spectrogram_predict, center_freq, bandwidth, center_freq_orig


def save_sample(colormapped_array, center_freq):
    cv2.imwrite('SAMPLES_LOW_POWER/SCANNER_SAMPLES_CF_' + str(center_freq) + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.jpg', colormapped_array)


def predict_samples(center_freq_recv, bandwidth_recv, num_center_frequencies_recv, overlap_recv, data, save_samples, memory_optimization):
    # Local prediction lists - thread-safe, no globals
    predicted_freq_list_4g = []
    predicted_freq_list_3g = []
    predicted_freq_list_2g = []

    center_freq_orig = center_freq_recv * emul
    bandwidth = bandwidth_recv * emul
    num_center_frequencies = num_center_frequencies_recv
    overlap = overlap_recv * emul

    tstart = datetime.now()

    # data is already sliced to [:, 357:1691] by caller
    spectrogram = data
    num_of_samples_in_freq = data.shape[0] // num_center_frequencies

    num_chunks = 1

    spectrogram_new = create_correct_spectrogram_by_rearranging_samples(num_center_frequencies, spectrogram, num_of_samples_in_freq)
    del spectrogram

    if memory_optimization == "YES":
        num_chunks, overlap = get_num_chunks_for_mem_optimization(bandwidth_recv, overlap)

    num_samples_in_chunk = int(MAX_AI_BANDWIDTH / num_khz_per_fft_point)

    for chunk_iterator in range(num_chunks):

        spectrogram_predict, center_freq, bandwidth, center_freq_orig = get_truncated_spectrum(spectrogram_new,
                num_chunks, chunk_iterator, center_freq_orig,
                num_samples_in_chunk, bandwidth, overlap)

        if memory_optimization == "YES" and num_chunks > 1:
            img = _normalizer.get_normalized_values(spectrogram_predict)
            del spectrogram_predict
        else:
            img = _normalizer.get_normalized_values(spectrogram_new)
            center_freq = center_freq_orig

        colormapped_array = _color_mapper.get_new_img(img)
        del img
        # colormap already returns uint8; just flip RGB->BGR for OpenCV/Ultralytics
        colormapped_array = colormapped_array[..., ::-1]

        xval_list = []
        chunk_start_indexes_in_new_image = []
        start_freq_for_each_chunk = []
        start_freq = center_freq - (bandwidth / 2)

        if save_samples == "YES":
            save_sample(colormapped_array, center_freq)

        # 3G/4G inference via Ultralytics (uses optimized C++ backend)
        with torch.no_grad():
            results_3g_4g = model_3g_4g_new(colormapped_array, conf=0.6, stream=True)

        xval_list = process_results_3g_4g(results_3g_4g, xval_list, start_freq, bandwidth,
                                          predicted_freq_list_3g, predicted_freq_list_4g)
        del results_3g_4g

        # Build 2G image from gaps between 3G/4G detections
        gap_slices = []
        for i in range(len(xval_list)):
            if i == 0:
                gap_slices.append(colormapped_array[:, 0:xval_list[i][0], :])
                chunk_start_indexes_in_new_image.append(0)
                start_freq_for_each_chunk.append(start_freq)
            else:
                start_freq_for_each_chunk.append(start_freq + (xval_list[i - 1][0] + xval_list[i - 1][1]) * 15000)
                gap_slices.append(colormapped_array[:, xval_list[i - 1][0] + xval_list[i - 1][1]:xval_list[i][0], :])

        if len(xval_list) != 0:
            gap_slices.append(colormapped_array[:, xval_list[i][0] + xval_list[i][1]:, :])
            start_freq_for_each_chunk.append(start_freq + (xval_list[i][0] + xval_list[i][1]) * 15000)
            colormapped_array_2G = np.concatenate(gap_slices, axis=1)
            chunk_start_indexes_in_new_image = []
            running_width = 0
            for sl in gap_slices:
                chunk_start_indexes_in_new_image.append(running_width)
                running_width += sl.shape[1]
        else:
            colormapped_array_2G = colormapped_array
            chunk_start_indexes_in_new_image.append(0)
            start_freq_for_each_chunk.append(start_freq)

        del gap_slices, colormapped_array

        # 2G inference via Ultralytics with actual image size (dynamic)
        with torch.no_grad():
            results_2g = model_2g_new(colormapped_array_2G, conf=0.3,
                                      imgsz=[colormapped_array_2G.shape[0], colormapped_array_2G.shape[1]],
                                      stream=True)

        process_results_2g(results_2g, start_freq_for_each_chunk, chunk_start_indexes_in_new_image, predicted_freq_list_2g)
        del results_2g, colormapped_array_2G

    tend = datetime.now()
    del spectrogram_new
    gc.collect()

    return (predicted_freq_list_4g, predicted_freq_list_3g, predicted_freq_list_2g, tend - tstart)


def recieve_samples(conn, initial_byte_size, scanner_ai_save_samples, memory_optimization):
    try:
        buf = conn.recv(initial_byte_size)
        scanner_ai_data_req = ai_model_pb2.AIModelReq()

        try:
            scanner_ai_data_req.ParseFromString(buf)
        except message.DecodeError as e:
            logging.error(f"Unable to parse incoming message: {buf}{e}")
            return

        scanner_ai_res = ai_model_pb2.AIModelRes()
        scanner_ai_res.predict_sample_res.result = ai_model_pb2.AIResult.AI_RESULT_SUCCESS_UNSPECIFIED
        scanner_ai_res.predict_sample_res.id = scanner_ai_data_req.predict_sample_req.id

        payload = scanner_ai_res.SerializeToString()
        conn.send(payload)

        if scanner_ai_data_req.WhichOneof("message") == "predict_sample_req":
            center_freq = scanner_ai_data_req.predict_sample_req.center_freq_khz
            bandwidth = scanner_ai_data_req.predict_sample_req.bw_khz
            num_center_freq = scanner_ai_data_req.predict_sample_req.num_chunks
            overlap = scanner_ai_data_req.predict_sample_req.overlay_khz
            sample_len = scanner_ai_data_req.predict_sample_req.samples_len
        else:
            logging.error("Wrong message type received expected predict_sample_req")
            return

        del scanner_ai_data_req

        # Pre-allocated receive buffer - avoids chunks list memory overhead
        recv_buf = bytearray(sample_len)
        view = memoryview(recv_buf)
        bytes_recd = 0
        while bytes_recd < sample_len:
            nbytes = conn.recv_into(view[bytes_recd:], min(sample_len - bytes_recd, 65000))
            if nbytes == 0:
                raise RuntimeError("Socket connection broken")
            bytes_recd += nbytes

        scanner_ai_data_req_1 = ai_model_pb2.AIModelReq()
        scanner_ai_data_req_1.ParseFromString(recv_buf)
        del recv_buf, view

        if scanner_ai_data_req_1.WhichOneof("message") == "sample_data_req":
            sample_data_req_id = scanner_ai_data_req_1.sample_data_req.id
            sample = np.array(scanner_ai_data_req_1.sample_data_req.samples, dtype=np.float32)
            del scanner_ai_data_req_1
            length = len(sample)
            sample = sample.reshape(length // fft_size, fft_size)
            # Extract the spectrogram slice as a contiguous copy so `sample` can be freed
            spectrogram_slice = np.array(sample[:, 357:1691])
            del sample
            predicted_4g, predicted_3g, predicted_2g, time_taken = predict_samples(center_freq, bandwidth, num_center_freq, overlap, spectrogram_slice, scanner_ai_save_samples, memory_optimization)
            del spectrogram_slice
        else:
            logging.error("Wrong message type received expected sample_data_req")
            del scanner_ai_data_req_1
            return

        logging.info(f"predicted 4G {predicted_4g}")
        scanner_ai_data_res = ai_model_pb2.AIModelRes()

        scanner_ai_data_res.sample_data_res.id = sample_data_req_id
        scanner_ai_data_res.sample_data_res.lte_freqs[:] = predicted_4g[:]
        scanner_ai_data_res.sample_data_res.umts_freqs[:] = predicted_3g[:]
        scanner_ai_data_res.sample_data_res.gsm_freqs[:] = predicted_2g[:]
        conn.send(scanner_ai_data_res.SerializeToString())

        logging.info("------------------HEADER CONTENTS-----------------")
        logging.info(f"center freq {center_freq}")
        logging.info(f"bandwidth {bandwidth}")
        logging.info(f"chunks {num_center_freq}")
        logging.info(f"overlap {overlap}")
        logging.info(f"sample len {sample_len}")

        logging.info("------------------AI MODEL PREDICTIONS-----------------")
        logging.info(f"Detected 4G frequencies by AI {predicted_4g}")
        logging.info(f"Detected 3G frequencies by AI {predicted_3g}")
        logging.info(f"Detected 2G frequencies by AI {predicted_2g}")

        logging.info(f"****************Total time taken by AI MODELS ****************************** {time_taken}")
        del sample, predicted_4g, predicted_3g, predicted_2g
        gc.collect()

    finally:
        conn.close()


def _worker_wrapper(conn, initial_byte_size, scanner_ai_save_samples, memory_optimization):
    _thread_semaphore.acquire()
    try:
        recieve_samples(conn, initial_byte_size, scanner_ai_save_samples, memory_optimization)
    finally:
        _thread_semaphore.release()


def main():
    scanner_ai_host = os.getenv('SCANNER_AI_IP', '0.0.0.0')
    scanner_ai_port = int(os.getenv('SCANNER_AI_PORT', 4444))
    scanner_ai_save_samples = str(os.getenv('SAVE_SAMPLES'))
    memory_optimization = str(os.getenv('MEM_OPTIMIZATION'))
    logging.info(f"Starting Scanner AI service listening on port {scanner_ai_port} {scanner_ai_host}...")
    s.bind((scanner_ai_host, scanner_ai_port))
    s.listen(1)
    while True:
        connection, _ = s.accept()
        reciever_thread = threading.Thread(target=_worker_wrapper, args=(connection, 1024, scanner_ai_save_samples, memory_optimization))
        reciever_thread.start()

if __name__ == "__main__":
    main()
