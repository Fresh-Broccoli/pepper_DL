from client2 import *
import requests
import os

def initiate_oc():
    return Client(model="ocsort", image_size=[640, 640], device="cuda", verbose=True, hand_raise_frames_thresh=3)

def initiate_bot():
    # BoTSORT default params
    args = bot_sort_make_parser().parse_args()
    args.ablation = False
    args.mot20 = not args.fuse_score

    return Client(model="botsort", image_size=[640, 640], device="cuda", verbose=True, args=args,
               hand_raise_frames_thresh=3)

def initiate_byte():
    args = byte_track_make_parser().parse_args()

    return Client(model="bytetrack", device="cuda", verbose=True, args=args,
               hand_raise_frames_thresh=3)


def oc_exp(draw = True, trial="distance", case="1m", shutdown=True, verbose=True, clear_img=False):
    p = os.path.join("exp_img", "OCSORT", trial, case)
    if not os.path.exists(p):
        os.makedirs(p)
    elif clear_img:
        print("Clearing old images...")
        for f in os.listdir(p):
            os.remove(os.path.join(p,f))
        print("Clearing old images successful!")
    # Used for both the distance and occlusion trial for OCSORT
    c = initiate_oc()

    # Main follow behaviour:
    try:
        c.experiment_follow(save_dir=p, draw=draw)
    except:
        if shutdown:
            c.shutdown()
        return c

def bot_exp(draw = True, trial="distance", case="1m", shutdown=True, clear_img=False):
    p = os.path.join("exp_img", "BoTSORT", trial, case)
    if not os.path.exists(p):
        os.makedirs(p)
    elif clear_img:
        print("Clearing old images...")
        for f in os.listdir(p):
            os.remove(os.path.join(p, f))
        print("Clearing old images successful!")
    # Used for both the distance and occlusion trial for OCSORT
    c = initiate_bot()
    # Main follow behaviour:
    c.experiment_follow(save_dir=p, draw=draw)

    if shutdown:
        c.shutdown()


def byte_exp(draw = True, trial="distance", case="1m", shutdown=True, clear_img=False):
    p = os.path.join("exp_img", "ByteTrack", trial, case)
    if not os.path.exists(p):
        os.makedirs(p)
    elif clear_img:
        print("Clearing old images...")
        for f in os.listdir(p):
            os.remove(os.path.join(p, f))
        print("Clearing old images successful!")
    # Used for both the distance and occlusion trial for OCSORT
    c = initiate_byte()

    c.experiment_follow(save_dir=p, draw=draw)

    if shutdown:
        c.shutdown()

def ocfollow():
    c = initiate_oc()
    #c = Client(image_size=[640, 640], device="cpu", max_age=60, verbose=True)
    # Main follow behaviour:
    c.follow_behaviour()
    # Must call
    c.shutdown()


def botfollow():
    c = initiate_bot()
    # Main follow behaviour:
    c.follow_behaviour()
    # Must call
    c.shutdown()


def livestream_camera_botsort():
    c = initiate_bot()
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
    c = initiate_oc()
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

def get_image_pred_test(client, image_no=60):
    start_time = time.time()
    for i in range(image_no):
        pred, img = client.predict(img=None, draw=True)
        #print(pred.shape)
        end_time = time.time() - start_time
    print(f"It took {str(end_time)} seconds to receive {str(image_no)} images and process them through YOLO-Pose + {client.model_name}. This means we were able to receive images from Pepper to server to client at {str(image_no/end_time)} FPS!")


def quick_shutdown():
    headers = {'content-type': "/setup/end"}
    response = requests.post("http://localhost:5000" + headers["content-type"], headers=headers)


if __name__ == "__main__":
    # BoTSORT default params
    #args = make_parser().parse_args()
    #args.ablation = False
    #args.mot20 = not args.fuse_score

    try:
        #c = initiate_oc()
        c = oc_exp(trial="occlusion", case="5m/1", clear_img=True)
        #bot_exp(trial="occlusion", case="5m", clear_img=True)
        #byte_exp(trial="occlusion", case="5m", clear_img=True)
    #oc = initiate_oc(trial="occlusion", case="5m")
    #bot = initiate_bot()
    #pred, img = bot.predict(None, draw=False)
    #print("pred:", pred)
    #print("pred type:", type(pred))
    #print("pred shape:", pred.shape)


    #bot = initiate_bot()
    #byte = initiate_byte()
    #for c in [oc, bot, byte]:
    #    get_image_pred_test(client = c)
    #get_image_pred_test(byte,)
    #oc.shutdown()
    except:
        print("Time it took for the robot to reach the target: ", c.end_time)
        quick_shutdown()