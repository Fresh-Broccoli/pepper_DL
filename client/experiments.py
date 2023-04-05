from client2 import *
import requests

def oc_exp():
    # Used for both the distance and occlusion trial for OCSORT
    c = Client(model="ocsort", image_size=[640, 640], device="cuda", verbose=True, hand_raise_frames_thresh=3)

    # Main follow behaviour:
    c.experiment_follow()
    # Must call
    c.shutdown()

def bot_exp():
    # BoTSORT default params
    args = bot_sort_make_parser().parse_args()
    args.ablation = False
    args.mot20 = not args.fuse_score

    # c = Client(model="botsort", image_size=[640,640], device="cuda", max_age=60, verbose=True, hand_raise_frames_thresh=3)
    c = Client(model="botsort", image_size=[640, 640], device="cuda", verbose=True, args=args,
               hand_raise_frames_thresh=3)
    # Main follow behaviour:
    c.experiment_follow()


def ocfollow():
    c = Client(model="ocsort", image_size=[640,640], device="cuda", max_age=60, verbose=False, hand_raise_frames_thresh=3)
    #c = Client(image_size=[640, 640], device="cpu", max_age=60, verbose=True)
    # Main follow behaviour:
    c.follow_behaviour()
    # Must call
    c.shutdown()


def botfollow():
    # BoTSORT default params
    args = bot_sort_make_parser().parse_args()
    args.ablation = False
    args.mot20 = not args.fuse_score

    # c = Client(model="botsort", image_size=[640,640], device="cuda", max_age=60, verbose=True, hand_raise_frames_thresh=3)
    c = Client(model="botsort", image_size=[640, 640], device="cuda", verbose=False, args=args,
               hand_raise_frames_thresh=3)
    # Main follow behaviour:
    c.follow_behaviour()
    # Must call
    c.shutdown()



def livestream_camera_botsort():
    c = Client(model="botsort", image_size=[640, 640], device="cuda", verbose=False, args=args,
               hand_raise_frames_thresh=3)
    vertical_offset = 0.5
    try:
        while True:
            pred, img = c.predict(img=None, draw=False)
            if len(pred) > 0:
                box = pred[0]
                img_shape = img.shape
                box_center = np.array([box[2] / 2 + box[0] / 2, box[1] * (1 - vertical_offset) + box[
                    3] * vertical_offset])  # box[1]/2+box[3]/2])
                frame_center = np.array((img_shape[1] / 2, img_shape[0] / 2))
                # diff = box_center - frame_center
                diff = frame_center - box_center
                horizontal_ratio = diff[0] / img_shape[1]
                vertical_ratio = diff[1] / img_shape[0]
                area = (box[2]-box[0])*(box[3]-box[1])
                area_ratio = area/(img_shape[0]*img_shape[1])
                print("BoT Prediction:", pred)
                print("Area ratio:", area_ratio)
                print("horizontal_ratio:", horizontal_ratio)
                print("vertical_ratio:", vertical_ratio)

    except Exception as e:
        print(e)
        c.shutdown()

def livestream_camera_ocsort():
    c = Client(model="ocsort", image_size=[640,640], device="cuda", max_age=60, verbose=False, hand_raise_frames_thresh=3)
    vertical_offset = 0.5
    try:
        while True:
            pred, img = c.predict(img=None, draw=False)
            if len(pred) > 0:
                box = pred[0]
                img_shape = img.shape
                box_center = np.array([box[2] / 2 + box[0] / 2, box[1] * (1 - vertical_offset) + box[
                    3] * vertical_offset])  # box[1]/2+box[3]/2])
                frame_center = np.array((img_shape[1] / 2, img_shape[0] / 2))
                # diff = box_center - frame_center
                diff = frame_center - box_center
                horizontal_ratio = diff[0] / img_shape[1]
                vertical_ratio = diff[1] / img_shape[0]
                area = (box[2]-box[0])*(box[3]-box[1])
                area_ratio = area/(img_shape[0]*img_shape[1])
                print("Prediction:", pred)
                print("Area ratio:", area_ratio)
                print("horizontal_ratio:", horizontal_ratio)
                print("vertical_ratio:", vertical_ratio)

    except Exception as e:
        print(e)
        c.shutdown()


def quick_shutdown():
    headers = {'content-type': "/setup/end"}
    response = requests.post("http://localhost:5000" + headers["content-type"], headers=headers)


if __name__ == "__main__":
    # BoTSORT default params
    #args = make_parser().parse_args()
    #args.ablation = False
    #args.mot20 = not args.fuse_score

    try:
        #oc_exp()
        bot_exp()
    except:
        quick_shutdown()