def height_estimate(distance, height):
    height = 1280 - height
    distance += 32
    camera_image_height = 1.48 * distance
    if height < 640:
        subject_height = 640 - height
        hf = 1.52 - (((subject_height*camera_image_height)/1280)/100)
    else:
        subject_height = height - 640
        hf = (((subject_height*camera_image_height)/1280)/100) + 1.52
    return hf