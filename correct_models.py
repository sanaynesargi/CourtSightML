import os
import shutil
import glob

def main():
  for player in os.listdir("./models"):
    path = f"./models/{player}"
    combined_name = player.split(" ")
    try:
      new_path = ""
      remove = True
      if len(combined_name) == 2:
        new_path = f"./models/{combined_name}"
      else:
        new_path = f"./models/{player}"
        combined_name = player
        remove = False
        
      #combined_name = combined_name[0] + combined_name[1]
      #shutil.copytree(path, f"./models/{combined_name}")
      
      h5 = glob.glob(f"{new_path}/trained_model/*.h5")[0]
      shutil.copy(h5, f"{new_path}/trained_model/{player}.h5")
      os.remove(h5)
      
      if remove:
        shutil.rmtree(path)
    except IndexError:
      continue


if __name__ == '__main__':
  main()