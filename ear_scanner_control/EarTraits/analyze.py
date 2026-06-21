import cv2
import glob
from EarTraits.scan_time_utils import PreImageAnalyzer, CamTrinsics
from typing import List, Optional
import pandas as pd
import os
from natsort import natsorted
import time

    
class CamPairGenerator:
   def __init__(self, ear_path: str, first_and_last_only=False):
      self.ear_path = ear_path
      self.paths = natsorted(glob.glob(ear_path + '/*'))
      self.pairs = self._match_pairs()
      self.first_and_last_only = first_and_last_only

   #@property
   def _match_pairs(self):
      """Generate pairs of matching images."""
      i=0
      for path in self.paths:
         if "_low_" in path:
            if self.first_and_last_only and 0< i < 49:
               i+=1
               continue
            i+=1
            low_path = path
            up_path = path.replace("_low_", "_up_")
            if not os.path.exists(up_path):
               raise FileNotFoundError(f"Matching image not found: {up_path}") 
            yield low_path, up_path
      
   def __iter__(self):
      return self.pairs
   
   def __len__(self):
      if self.first_and_last_only:
         return 2
      else:
         return len(self.paths) // 2
   
def run_analysis(path_lst: List, DOVOXELCARVING, DOKERNELCOUNT, DOSAM, stopevent):

   INTRINSIC_P = "/home/mais/Desktop/EarScanJetson/ScannerCamSettings.csv"
   if not os.path.exists(INTRINSIC_P):
      raise FileNotFoundError(INTRINSIC_P)
   REALWORLD_POINTS_P = "/home/mais/Desktop/EarScanJetson/Cps2401_pluscorners.csv"
   if not os.path.exists(REALWORLD_POINTS_P):
      raise FileNotFoundError(REALWORLD_POINTS_P)
   low_errors = None
   up_errors = None

   camtrins_low = CamTrinsics(INTRINSIC_P, REALWORLD_POINTS_P, campos_n=2)
   camtrins_up = CamTrinsics(INTRINSIC_P, REALWORLD_POINTS_P, campos_n=2)
  
   for p in path_lst:
      if stopevent.is_set():
         break
      cam_paths = CamPairGenerator(p, first_and_last_only=True)
      scantime_low = PreImageAnalyzer(camtrins_low)
      scantime_up = PreImageAnalyzer(camtrins_up)
      for i, (low_path, up_path) in enumerate(cam_paths):
         print(f"Position {i}/{len(cam_paths)}")
         scantime_low(low_path)
         scantime_up(up_path)
      low_errors = scantime_low.err_lst
      up_errors = scantime_up.err_lst
      
      scantime_low.reset_trinsics()
      scantime_up.reset_trinsics()
      if stopevent.is_set():
            low_errors.append(f"Programme was stopped after {p}")
            break 
      print("Ear done")
   del scantime_low 
   del scantime_up
   del camtrins_low
   del camtrins_up
   return low_errors, up_errors
 

      
   
  

    


  