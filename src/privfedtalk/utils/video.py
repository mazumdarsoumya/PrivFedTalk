import torch
def center_crop(video: torch.Tensor, size: int):
    if video.dim()==4:
        T,C,H,W=video.shape; top=(H-size)//2; left=(W-size)//2
        return video[:,:,top:top+size,left:left+size]
    if video.dim()==5:
        B,T,C,H,W=video.shape; top=(H-size)//2; left=(W-size)//2
        return video[:,:,:,top:top+size,left:left+size]
    raise ValueError("Unexpected shape")
