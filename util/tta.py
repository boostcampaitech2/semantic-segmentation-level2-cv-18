import torch


def vertical_flip(images):
    flip_images = torch.flip(images, dims=[-1])
    return flip_images


def horizontal_flip(images):
    flip_images = torch.flip(images, dims=[-2])
    return flip_images


def get_tta_list(cfg):
    tta_cfg = cfg["EXPERIMENTS"]["TTA"]
    tta_list = []

    if tta_cfg["TURN_ON"]:
        tta_apply_candidates = tta_cfg["AVAILABLE_LIST"]
        if tta_apply_candidates["VERTICAL_FLIP_TURN_ON"]:
            tta_list.append(vertical_flip)
        if tta_apply_candidates["HORIZONTAL_FLIP_TURN_ON"]:
            tta_list.append(horizontal_flip)

    return tta_list


# Only used for test.
def main():
    cfg = {
        "EXPERIMENTS": {
            "TTA": {
                "TURN_ON": True,
                "AVAILABLE_LIST": {
                    "VERTICAL_FLIP_TURN_ON": True,
                    "HORIZONTAL_FLIP_TURN_ON": True,
                },
            }
        }
    }

    print(get_tta_list(cfg))
    return


# Only used for test.
if __name__ == "__main__":
    main()
