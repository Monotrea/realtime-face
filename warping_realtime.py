import dlib
import cv2
import numpy as np
import time
import sys

lk_params = dict( winSize  = (20, 20),
                maxLevel = 10,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                minEigThreshold= 1.0 * 10**(-2))

def get_face_rect(detector, frame, opt=1):
    ret = None
    dets = detector(frame, 0)
    if len(dets) == 0:
        print('no face')
        return ret
    if opt == 1:
        print('face detected!')
    ret = dets[0]
    return ret

def get_face_size(rect):
    face_width = rect.right() - rect.left()
    face_height = rect.bottom() - rect.top()
    return face_width, face_height

def get_facepoints(predictor, frame, rect):
    facepoints = []
    shape = predictor(frame, rect)
    for i in range(len(shape.parts())):
        x = shape.part(i).x
        y = shape.part(i).y
        facepoints.append((int(x), int(y)))
    return facepoints

def append_8points(original_facepoints, framewidth, frameheight):
    facepoints = list(original_facepoints)
    facepoints.append((0,0))
    facepoints.append((0, int(frameheight/2)))
    facepoints.append((0, frameheight - 1))
    facepoints.append((int(framewidth/2), 0))
    facepoints.append((framewidth - 1, 0))
    facepoints.append((framewidth - 1, int(frameheight/2)))
    facepoints.append((framewidth - 1, frameheight - 1))
    facepoints.append((int(framewidth/2), frameheight - 1))
    return facepoints

def get_delaunay_triangles(points, framewidth, frameheight):
    framerect = (0, 0, framewidth, frameheight)
    framesubdiv = cv2.Subdiv2D(framerect)
    for p in points:
        framesubdiv.insert(p)
    frametriangle = framesubdiv.getTriangleList()
    frametriangleindex = []
    frametriangleposition = []
    for triangle_points in frametriangle:
        temporary_indexlist = []
        triangle_points_list = []
        for count in range(0, 3):
            result_index = -1
            for temp_index, temp_points in enumerate(points):
                if (abs(triangle_points[2 * count] - temp_points[0]) < 1.0) and (abs(triangle_points[2 * count + 1] - temp_points[1]) < 1.0):
                    result_index = temp_index
            if result_index != -1:
                temporary_indexlist.append(result_index)
            triangle_points_list.append((triangle_points[2 * count], triangle_points[2 * count + 1]))
        if (len(temporary_indexlist) == 3):
            frametriangleindex.append((temporary_indexlist[0], temporary_indexlist[1], temporary_indexlist[2]))
            frametriangleposition.append(triangle_points_list)
    return frametriangleindex, frametriangleposition

def getOpticalTrackPoints(points, fix_points, old_gray, frame_gray, countframe=0):
    return_points = []
    result_points, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, points, None, **lk_params)
    result_points = np.reshape(result_points, (-1, 2))
    for temp_index, stval in enumerate(st):
        if stval == 1:
            if np.linalg.norm(np.array(result_points[temp_index]) - np.array(points[temp_index])) > 1.0:
                return_points.append(fix_points[temp_index])
            else:
                return_points.append(tuple(result_points[temp_index]))
        else:
            return_points.append(fix_points[temp_index])
    return return_points

def get_move_vector(points, default_points, input_facewidth, input_faceheight, drive_facewidth, drive_faceheight):
    move_vector = np.array(points) - default_points
    sc = np.array([[input_facewidth / drive_facewidth, 0], [0, input_faceheight / drive_faceheight]])
    move_vector = move_vector.dot(sc)
    return np.array(move_vector)

def apply_move_vector(input_facepoints, move_vector):
    moved_points = np.array(input_facepoints)
    moved_points = moved_points + move_vector
    return moved_points

def get_triangle_points(triangle_indexlist, points):
    result_points = []
    for index1, index2, index3 in triangle_indexlist:
        result_points.append(np.array([points[index1], points[index2], points[index3]]))
    return result_points

def get_rect_relative_int_points(rect, points):
    result = np.array(points) - np.array(rect[:2])
    result = np.int32(result)
    return result

def get_rect_relative_float_points(rect, points):
    result = np.array(points) - np.array(rect[:2])
    result = np.float32(result)
    return result

def get_black_img(width, height):
    return np.zeros((height, width, 3), np.uint8)

def trim_pict(frame, x, y, width, height):
    return np.array(frame[y:(y + height), x:(x + width)], np.uint8)

def position2intpoints(pos):
    points = np.array(pos, np.int32)
    points = np.reshape(points, (-1, 1, 2))
    return points

def position2floatpoints(pos):
    points = np.array(pos, np.float32)
    points = np.reshape(points, (-1, 1, 2))
    return points

def AffineTransform(input_frame, input_facepoints, default_facepoints, now_facepoints, input_facewidth, input_faceheight, default_width, default_height, input_delaunay_index, input_delaunay_points, input_frame_width, input_frame_height):
    move_vector = get_move_vector(now_facepoints, default_facepoints, input_facewidth, input_faceheight, default_width, default_height)
    moved_points = apply_move_vector(input_facepoints, move_vector)
    trianglepoints_after = get_triangle_points(input_delaunay_index, moved_points)
    dest_image = get_black_img(input_frame_width, input_frame_height)
    for temp_index2, pts in enumerate(trianglepoints_after):
        input_points = position2floatpoints(input_delaunay_points[temp_index2])
        output_points = position2floatpoints(pts)
        input_rect = cv2.boundingRect(input_points)
        output_rect = cv2.boundingRect(output_points)
        if output_rect[0] < 0 or output_rect[1] < 0 or (output_rect[0] + output_rect[2] > input_frame_width) or (output_rect[1] + output_rect[3] > input_frame_height):
            return False, dest_image
        input_relative_points = get_rect_relative_float_points(input_rect, input_delaunay_points[temp_index2])
        output_relative_points = get_rect_relative_float_points(output_rect, pts)
        output_relative_intpoints = get_rect_relative_int_points(output_rect, pts)
        input_crop = trim_pict(input_frame, input_rect[0], input_rect[1], input_rect[2], input_rect[3])
        output_crop = get_black_img(output_rect[2], output_rect[3])
        warping_matrix = cv2.getAffineTransform(input_relative_points, output_relative_points)
        cv2.warpAffine(input_crop, warping_matrix, (output_rect[2], output_rect[3]), output_crop, cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101)
        mask_frame = get_black_img(output_rect[2], output_rect[3])
        cv2.fillConvexPoly(mask_frame, output_relative_intpoints, (1, 1, 1), 16, 0)
        cv2.multiply(output_crop, mask_frame, output_crop)
        screen = np.ones_like(mask_frame)
        cv2.multiply(dest_image[output_rect[1]:(output_rect[1] + output_rect[3]), output_rect[0]:(output_rect[0] + output_rect[2])], screen - mask_frame, dest_image[output_rect[1]:(output_rect[1] + output_rect[3]), output_rect[0]:(output_rect[0] + output_rect[2])])
        dest_image[output_rect[1]:(output_rect[1] + output_rect[3]), output_rect[0]:(output_rect[0] + output_rect[2])] += output_crop
    return True, dest_image

def main():
    if len(sys.argv) == 1:
        print('Please input a face picture file.')
        exit()
    #define common value
    input_picture_name = sys.argv[1]
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    #input_frame_value
    input_frame = cv2.imread(input_picture_name)
    input_frame_height, input_frame_width = input_frame.shape[:2]
    #input_video_value
    input_video = cv2.VideoCapture(0)
    input_video_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_video_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_video_fps = input_video.get(cv2.CAP_PROP_FPS)
    input_video_framecount = input_video.get(cv2.CAP_PROP_FRAME_COUNT)
    #input_face_property
    print('Recognizing a face in the input picture...')
    input_facerect = get_face_rect(detector, input_frame)
    input_facewidth, input_faceheight = get_face_size(input_facerect)
    input_facepoints = append_8points(get_facepoints(predictor, input_frame, input_facerect), input_frame_width, input_frame_height)
    input_delaunay_index, input_delaunay_points = get_delaunay_triangles(input_facepoints, input_frame_width, input_frame_height)
    print('done')
    #default_face_point_property
    default_facepoints = []
    default_height = 0
    default_width = 0
    #facepoints list of input_video
    input_video_frames = []
    input_video_facepoints = []
    #for_created_frames
    created_frames = []
    #count of frames loaded
    countframe = 0
    old_gray = []
    old_frame = np.array((0))
    old_points = np.array((0))
    k = cv2.waitKey(0) & 0xFF
    lf = get_black_img(input_video_width, input_video_height)
    name1 = 'press any key except 0 unless your face is neutral and then press 0 to next'
    cv2.imshow(name1, lf)
    cv2.moveWindow(name1, 0, 0)
    while (input_video.isOpened()):
        ret, lf = input_video.read()
        cv2.imshow(name1, lf)
        k = cv2.waitKey(1) & 0xFF
        if k == 48:
            break
    cv2.destroyWindow(name1)

    name2 = 'press 1 to finish'
    newframe = get_black_img(input_frame_width, input_frame_height)
    cv2.imshow(name2, newframe)
    cv2.moveWindow(name2, 0, 0)

    start_time = time.time()
    time_wasted = 0.0
    while(input_video.isOpened()):
        del_time_start = time.time()
        countframe += 1

        ret, loadframe = input_video.read()
        if ret == False:
            break

        if countframe == 1:
            old_frame = loadframe

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(loadframe, cv2.COLOR_BGR2GRAY)
        old_fix_frame = old_frame
        old_frame = loadframe

        face_rect = get_face_rect(detector, loadframe, 0)
        if face_rect == None:
            time_wasted += time.time() - del_time_start
            old_frame = old_fix_frame
            countframe -= 1
            continue
        (face_height, face_width) = get_face_size(face_rect)
        face_points = get_facepoints(predictor, loadframe, face_rect)
        if (countframe == 1):
            old_points = position2floatpoints(face_points)
        face_points = getOpticalTrackPoints(old_points, face_points, old_gray, frame_gray, countframe)
        old_fix = np.array(old_points)
        old_points = position2floatpoints(face_points)
        face_points = append_8points(face_points, input_video_width, input_video_height)
        if (countframe == 1):
            default_facepoints = face_points
            default_width = face_width
            default_height = face_height
        ret, newframe = AffineTransform(input_frame, input_facepoints, default_facepoints, face_points, input_facewidth, input_faceheight, default_width, default_height, input_delaunay_index, input_delaunay_points, input_frame_width, input_frame_height)
        if ret == False:
            time_wasted += time.time() - del_time_start
            old_frame = old_fix_frame
            old_points = old_fix
            countframe -= 1
            continue
        #for points_video
        input_video_frames.append(loadframe)
        input_video_facepoints.append(face_points)
        #
        created_frames.append(newframe)
        cv2.imshow(name2, newframe)
        k = cv2.waitKey(1) & 0xFF
        if k == 49:
            break

    finish_time = time.time()
    cv2.destroyAllWindows()
    time_consumed = finish_time - start_time - time_wasted
    output_fps = countframe / time_consumed

    output_videofourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('output.mp4', output_videofourcc, output_fps, (input_frame_width, input_frame_height))
    for frame in created_frames:
        output_video.write(frame)
    output_video.release()
    pointsvideo_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    pointsvideo = cv2.VideoWriter('pointsvideo.mp4', pointsvideo_fourcc, output_fps, (input_video_width, input_video_height))
    for temp_index, frame in enumerate(input_video_frames):
        #get points frame
        for pind, (x, y) in enumerate(input_video_facepoints[temp_index]):
            cv2.circle(frame, (int(x), int(y)), 2, (pind * 3, 255, pind * 3), 5)
        pointsvideo.write(frame)
    input_video.release()
    pointsvideo.release()


if __name__ == '__main__':
    main()