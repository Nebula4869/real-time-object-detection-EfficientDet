from efficientdet.model import EfficientDetBackbone
from torch.backends import cudnn
import numpy as np
import torchvision
import torch
import time
import cv2

INPUT_SIZES = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4

cudnn.fastest = True
cudnn.benchmark = True


def bbox_transform(anchors, regression):
    y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
    x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
    ha = anchors[..., 2] - anchors[..., 0]
    wa = anchors[..., 3] - anchors[..., 1]

    w = regression[..., 3].exp() * wa
    h = regression[..., 2].exp() * ha

    y_centers = regression[..., 0] * ha + y_centers_a
    x_centers = regression[..., 1] * wa + x_centers_a

    ymin = y_centers - h / 2.
    xmin = x_centers - w / 2.
    ymax = y_centers + h / 2.
    xmax = x_centers + w / 2.

    return torch.stack([xmin, ymin, xmax, ymax], dim=2)


def clip_boxes(boxes, img):
    batch_size, num_channels, height, width = img.shape

    boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
    boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

    boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
    boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)

    return boxes


def realtime_detection(compound_coef, camera_id, model_file_path):

    input_size = INPUT_SIZES[compound_coef]

    classnames = []
    with open('coco.names') as f:
        for name in f:
            classnames.append(name.split('\n')[0])

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(classnames))
    model.load_state_dict(torch.load(model_file_path))
    model.requires_grad_(False)
    model.eval()
    model = model.cuda()

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while cap.isOpened():
        start = time.time()

        _, frame = cap.read()
        mean = np.array([0.406, 0.456, 0.485])
        std = np.array([0.225, 0.224, 0.229])
        normalized_img = (frame / 255 - mean) / std

        old_h, old_w, channel = normalized_img.shape
        if old_w > old_h:
            new_w = input_size
            new_h = input_size * old_h // old_w
        else:
            new_w = input_size * old_w // old_h
            new_h = input_size
    
        canvas = np.zeros((input_size, input_size, channel), np.float32)
        canvas[:new_h, :new_w] = cv2.resize(normalized_img, (new_w, new_h))
    
        x = torch.stack([torch.from_numpy(canvas).cuda()], 0)
        x = x.to(torch.float32).permute(0, 3, 1, 2)

        with torch.no_grad():
            features, regression, classification, anchors = model(x)
    
            transformed_anchors = bbox_transform(anchors, regression)
            transformed_anchors = clip_boxes(transformed_anchors, x)
            scores = torch.max(classification, dim=2, keepdim=True)[0]
            scores_over_thresh = (scores > SCORE_THRESHOLD)[:, :, 0]
    
            if scores_over_thresh[0].sum() == 0:
                out = {
                    'rois': np.array(()),
                    'class_ids': np.array(()),
                    'scores': np.array(()),
                }
            else:
                classification_per = classification[0, scores_over_thresh[0, :], ...].permute(1, 0)
                transformed_anchors_per = transformed_anchors[0, scores_over_thresh[0, :], ...]
                scores_per = scores[0, scores_over_thresh[0, :], ...]
                scores_, classes_ = classification_per.max(dim=0)
                anchors_nms_idx = torchvision.ops.boxes.batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=IOU_THRESHOLD)
        
                if anchors_nms_idx.shape[0] != 0:
                    classes_ = classes_[anchors_nms_idx]
                    scores_ = scores_[anchors_nms_idx]
                    boxes_ = transformed_anchors_per[anchors_nms_idx, :]
        
                    out = {
                        'rois': boxes_.cpu().numpy(),
                        'class_ids': classes_.cpu().numpy(),
                        'scores': scores_.cpu().numpy(),
                    }
                else:
                    out = {
                        'rois': np.array(()),
                        'class_ids': np.array(()),
                        'scores': np.array(()),
                    }

        if len(out['rois']) != 0:
            out['rois'][:, [0, 2]] = out['rois'][:, [0, 2]] / (new_w / old_w)
            out['rois'][:, [1, 3]] = out['rois'][:, [1, 3]] / (new_h / old_h)
        
        for i in range(len(out['rois'])):
            left, top, right, bottom = out['rois'][i].astype(np.int)
            score = float(out['scores'][i])
            classname = classnames[out['class_ids'][i]]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, '{}: {:.2f}'.format(classname, score), (left, top), cv2.FONT_HERSHEY_DUPLEX, (right - left) / 250, (0, 255, 0), 1)
    
        fps = 1 / (time.time() - start)
        cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (0, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)

        cv2.imshow('', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    realtime_detection(0, 1, 'models/efficientdet-d0.pth')
