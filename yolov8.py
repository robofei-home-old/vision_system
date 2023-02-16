import cv2
import pyzed.sl as sl
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


class DetectionPredictor(BasePredictor):
    cfg=DEFAULT_CFG
    model = cfg.model
    print("iniciando programa")
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))
    print("iniciando pre processamento de imagem")

    def preprocess(self, img):
        print("1")
        print(type(img))
        img = torch.from_numpy(img).to(self.model.device)
        print("2")
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        print("3")
        img /= 255  # 0 - 255 to 0.0 - 1.0
        print("4")
        return img
    print("iniciando posprocessamento de imagem")

    def postprocess(self, preds, img, orig_img, classes=None):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            results.append(Results(boxes=pred, orig_shape=shape[:2]))
        return results
    print("iniciando escrita dos resultados")

    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes  # TODO: make boxes inherit from tensors
        if len(det) == 0:
            return log_string
        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # write
        for d in reversed(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            if self.args.save_txt:  # Write to file
                line = (cls, *(d.xywhn.view(-1).tolist()), conf) \
                    if self.args.save_conf else (cls, *(d.xywhn.view(-1).tolist()))  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                name = f"id:{int(d.id.item())} {self.model.names[c]}" if d.id is not None else self.model.names[c]
                label = None if self.args.hide_labels else (name if self.args.hide_conf else f'{name} {conf:.2f}')
                self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if self.args.save_crop:
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string

print("iniciando segunda parte do programa")
def predict_zed(cfg=DEFAULT_CFG):    # Create a ZED camera object
    model = cfg.model or "yolov8n.pt"
    zed = sl.Camera()
    print("criacao do objeto da zed finalizado")
    print("iniciando configuracao da zed")
    # Set the configuration parameters for the ZED camera
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    print("configuracao da zed concluida")
    # Open the ZED camera
    print("tentando estabelecer conexao com a zed...")
    print("tentando estabelecer conexao com a zed...")
    print("tentando estabelecer conexao com a zed...")

    if not zed.is_opened():
        print("Opening ZED Camera...")
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
    print("conexao com a zed estabelecida")
    # Create a detection predictor object
    predictor = DetectionPredictor()

    # Set the source of the predictor to the ZED camera feed

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()

    while True:
        # Capture a new frame from the ZED camera
        try:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                # Retrieve the left image of the ZED camera and convert it to a numpy array
                zed.retrieve_image(mat, sl.VIEW.LEFT_UNRECTIFIED)
                left_image = mat.get_data()
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite("teste.jpg", left_image)
                print("got here!!!")

                # Preprocess the image
                #img = predictor.preprocess(left_image)

                # Run object detection on the preprocessed image
                #preds = predictor.model(img)

                # Postprocess the detection results
                #results = predictor.postprocess(preds, img, left_image)

                # Annotate the image with the detection results
                #annotator = predictor.get_annotator(left_image)
                #for result in results:
                #    annotator.annotate(result)

                # Call the `predict_cli` method of the `DetectionPredictor` class with the modified `args` dictionary
                args = dict(model=model, source="teste.jpg")
                predictor = DetectionPredictor(overrides=args)
                x = predictor.predict_cli()

                # Display the annotated image
                cv2.imshow("Annotated Image", x)

                # Exit the loop if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except:
            zed.close()
        # Retrieve the left image of the ZED camera and convert it to a numpy array

    # Close the ZED camera
    zed.close()
    cv2.destroyAllWindows()

predict_zed()