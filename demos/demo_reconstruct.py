# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points
from decalib.models.lbs import batch_rodrigues

def extract_camera_parameters(codedict, opdict, image_size=224):
    """
    Extract comprehensive camera parameters from DECA output.
    
    Args:
        codedict: Dictionary containing encoded DECA parameters
        opdict: Dictionary containing decoded DECA output
        image_size: Input image size (default 224)
    
    Returns:
        Dictionary containing camera parameters in multiple formats
    """
    # Extract raw camera parameters [scale, tx, ty]
    cam_params = codedict['cam'][0].cpu().numpy()  # [3]
    scale = float(cam_params[0])
    tx = float(cam_params[1])
    ty = float(cam_params[2])
    
    # Extract pose parameters (global head rotation in axis-angle format)
    pose_params = codedict['pose'][0].cpu().numpy()  # [6]
    global_rot_axis_angle = pose_params[:3]  # First 3 are global rotation
    jaw_pose_axis_angle = pose_params[3:]    # Last 3 are jaw pose
    
    # Convert axis-angle to rotation matrix
    global_rot_axis_angle_torch = torch.from_numpy(global_rot_axis_angle).float().unsqueeze(0)
    rotation_matrix = batch_rodrigues(global_rot_axis_angle_torch)[0].cpu().numpy()
    
    # Convert rotation matrix to Euler angles (in radians)
    def rotation_matrix_to_euler_angles(R):
        """Convert rotation matrix to Euler angles (XYZ convention)."""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.array([x, y, z])
    
    euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)
    
    # Compute camera location in world space
    # In DECA's weak perspective model, the camera is positioned along -Z axis
    # The distance is inversely related to the scale parameter
    # Typical scale of ~8 corresponds to a standard viewing distance
    camera_distance = 10.0 / scale if scale > 0 else 10.0
    
    # Camera location (assuming camera looks down -Z axis at origin)
    # Apply inverse rotation to get camera position in world space
    camera_location_camera_space = np.array([0, 0, camera_distance])
    camera_location_world = rotation_matrix.T @ camera_location_camera_space
    
    # Camera look direction (inverse of camera's forward vector)
    camera_forward = rotation_matrix @ np.array([0, 0, -1])
    camera_up = rotation_matrix @ np.array([0, 1, 0])
    
    # Build comprehensive camera dict
    camera_dict = {
        # Raw DECA parameters
        "deca_cam_params": {
            "scale": scale,
            "tx": tx,
            "ty": ty
        },
        
        # Head pose (NOT camera rotation, but head rotation)
        "head_rotation": {
            "axis_angle": global_rot_axis_angle.tolist(),
            "rotation_matrix": rotation_matrix.tolist(),
            "euler_angles_rad": euler_angles.tolist(),
            "euler_angles_deg": np.degrees(euler_angles).tolist()
        },
        
        # Jaw pose
        "jaw_pose": {
            "axis_angle": jaw_pose_axis_angle.tolist()
        },
        
        # Camera extrinsics (derived)
        "camera_extrinsics": {
            "location": camera_location_world.tolist(),
            "rotation_matrix": rotation_matrix.tolist(),
            "look_at": (camera_location_world - camera_forward).tolist(),
            "up_vector": camera_up.tolist(),
            "forward_vector": camera_forward.tolist(),
            "camera_distance": camera_distance
        },
        
        # Projection parameters
        "projection": {
            "type": "orthographic",
            "scale": scale,
            "image_size": image_size,
            "translation_2d": [tx, ty]
        },
        
        # Additional metadata
        "notes": {
            "coordinate_system": "FLAME canonical space with Y-up",
            "camera_model": "weak perspective (orthographic)",
            "head_rotation_note": "Head rotation is in FLAME space, not camera rotation",
            "scale_interpretation": "Larger scale means face appears larger (camera closer)",
            "translation_note": "2D translation in normalized image coordinates [-1, 1]"
        }
    }
    
    # Add vertices bounding box for reference
    if 'verts' in opdict:
        verts = opdict['verts'][0].cpu().numpy()
        bbox_min = verts.min(axis=0).tolist()
        bbox_max = verts.max(axis=0).tolist()
        bbox_center = ((verts.min(axis=0) + verts.max(axis=0)) / 2).tolist()
        
        camera_dict["mesh_info"] = {
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "bbox_center": bbox_center,
            "num_vertices": int(verts.shape[0])
        }
    
    return camera_dict

def inference_deca(images_dir='TestSamples/examples', 
         output_dir='TestSamples/examples/results', 
         device='cuda', 
         iscrop=True, 
         sample_step=10, 
         detector='fan', 
         rasterizer_type='pytorch3d', 
         render_orig=True, 
         freezeHeadPose=False, 
         useTex=False, 
         extractTex=True, 
         saveVis=True, 
         saveKpt=True, 
         saveDepth=False, 
         saveObj=False, 
         saveMat=False, 
         saveImages=False):
    # if rasterizer_type != 'standard':
    #     render_orig = False
    os.makedirs(output_dir, exist_ok=True)

    # load test images 
    testdata = datasets.TestData(images_dir, iscrop=iscrop, face_detector=detector, sample_step=sample_step)

    # run DECA
    deca_cfg.model.use_tex = useTex
    deca_cfg.rasterizer_type = rasterizer_type
    deca_cfg.model.extract_tex = extractTex
    deca = DECA(config = deca_cfg, device=device)
    # for i in range(len(testdata)):
    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename']
        sample_dir = os.path.join(output_dir, name)
        os.makedirs(sample_dir, exist_ok=True)
        images = testdata[i]['image'].to(device)[None,...]
        with torch.no_grad():
            codedict = deca.encode(images)
            codedict = deca.apply_head_pose_lock(codedict, freeze_head_pose=freezeHeadPose)
            opdict, visdict = deca.decode(codedict) #tensor
            if render_orig:
                tform = testdata[i]['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1,2).to(device)
                original_image = testdata[i]['original_image'][None, ...].to(device)
                _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)    
                orig_visdict['inputs'] = original_image            
            flame_param_dict = deca.export_flame_parameters(codedict)

        if flame_param_dict:
            param_path = os.path.join(sample_dir, f'{name}_flame_params.npz')
            np.savez(param_path, **flame_param_dict)
        
        # Save camera parameters as JSON
        camera_params = extract_camera_parameters(codedict, opdict, image_size=deca_cfg.dataset.image_size)
        camera_json_path = os.path.join(sample_dir, f'{name}_camera.json')
        with open(camera_json_path, 'w') as f:
            json.dump(camera_params, f, indent=2)
        
        # -- save results
        if saveDepth:
            depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
            visdict['depth_images'] = depth_image
            cv2.imwrite(os.path.join(sample_dir, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
        if saveKpt:
            np.savetxt(os.path.join(sample_dir, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
            np.savetxt(os.path.join(sample_dir, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
        if saveObj:
            deca.save_obj(os.path.join(sample_dir, name + '.obj'), opdict)
        if saveMat:
            opdict_np = util.dict_tensor2npy(opdict)
            for param_name, param_value in flame_param_dict.items():
                opdict_np[f'flame_{param_name}'] = param_value
            savemat(os.path.join(sample_dir, name + '.mat'), opdict_np)
        if saveVis:
            cv2.imwrite(os.path.join(output_dir, name + '_vis.jpg'), deca.visualize(visdict))
            if render_orig:
                cv2.imwrite(os.path.join(output_dir, name + '_vis_original_size.jpg'), deca.visualize(orig_visdict))
        if saveImages:
            for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images', 'landmarks2d']:
                if vis_name not in visdict.keys():
                    continue
                image = util.tensor2image(visdict[vis_name][0])
                cv2.imwrite(os.path.join(sample_dir, name + '_' + vis_name +'.jpg'), util.tensor2image(visdict[vis_name][0]))
                if render_orig:
                    image = util.tensor2image(orig_visdict[vis_name][0])
                    cv2.imwrite(os.path.join(sample_dir, 'orig_' + name + '_' + vis_name +'.jpg'), util.tensor2image(orig_visdict[vis_name][0]))
    print(f'-- please check the results in {output_dir}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--images_dir', default='TestSamples/examples', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--output_dir', default='TestSamples/examples/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    parser.add_argument('--freezeHeadPose', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='set true to keep the global head orientation fixed during reconstruction')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    
    args = parser.parse_args()
    
    inference_deca(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        device=args.device,
        iscrop=args.iscrop,
        sample_step=args.sample_step,
        detector=args.detector,
        rasterizer_type=args.rasterizer_type,
        render_orig=args.render_orig,
        freezeHeadPose=args.freezeHeadPose,
        useTex=args.useTex,
        extractTex=args.extractTex,
        saveVis=args.saveVis,
        saveKpt=args.saveKpt,
        saveDepth=args.saveDepth,
        saveObj=args.saveObj,
        saveMat=args.saveMat,
        saveImages=args.saveImages
    )