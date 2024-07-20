import torch
import sweep.common.training as common
from prettytable import PrettyTable

from sweep.ATT_BH.model import CI3P_ATT_BH
from sweep.CVAE.model import CI3PP_CVAE
from sweep.CVAE_ATT.model import CI3PP_CVAE_ATT
from sweep.m2p3.model import M2P3
from sweep.p3vi.model import P3VI

data_paths = [
    './P3VI/data/single/01_int.npy',
#     './P3VI/data/single/02_int.npy',
#     './P3VI/data/single/03_int.npy',
#     './P3VI/data/single/04_int.npy',
#     './P3VI/data/single/05_int.npy',
#     './P3VI/data/single/06_int.npy',
]

# data_paths = [
    # './P3VI/data/car_dump/04_int.npy',
    # './P3VI/data/single/04_int.npy',
# ]

# data_paths = [
#     './P3VI/data/single/01_non_int.npy',
#     './P3VI/data/single/02_non_int.npy',
#     './P3VI/data/single/03_non_int.npy',
#     './P3VI/data/single/04_non_int.npy',
#     './P3VI/data/single/05_non_int.npy',
#     './P3VI/data/single/06_non_int.npy',
# ]

m2p3_mse = []
m2p3_fde = []

p3vi_mse = []
p3vi_fde = []


ci3pp_ATT_mse = []
ci3pp_ATT_fde = []

ci3pp_ATT_SH_mse = []
ci3pp_ATT_SH_fde = []

ci3pp_ATT_BH_mse = []
ci3pp_ATT_BH_fde = []

ci3pp_CVAE_mse = []
ci3pp_CVAE_fde = []

ci3pp_CVAE_ATT_mse = []
ci3pp_CVAE_ATT_fde = []

n_obs = 60
n_pred = 80

for p in data_paths:
    print(20*"#")
    print(p)

    # print("M2P3")
    # m2p3 = M2P3(predict_frames=n_pred)
    # m2p3.cuda()
    # m2p3.load_state_dict(torch.load("./_out/m2p3/obs60_pred80/_batch1024_lr8.892721064394223e-05/2024-06-18_18-05-11/model__batch1024_lr8.892721064394223e-05.pth"))
    # mse, fde = common.test(
    #     path=p,
    #     n_obs=n_obs,
    #     n_pred=n_pred,
    #     batch_size=1024,
    #     model=m2p3,
    #     is_cvae=True,
    #     is_m2p3=True
    # )
    # m2p3_mse.append(mse)
    # m2p3_fde.append(fde)
    # print(20 * "#", "\n")
    #
    # print("P3VI")
    # p3vi = P3VI(n_predict_frames=n_pred)
    # p3vi.cuda()
    # p3vi.load_state_dict(torch.load("./_out/p3vi/obs60_pred80/_batch256_lr6.621334794064622e-05/2024-06-17_21-04-16/model__batch256_lr6.621334794064622e-05.pth"))
    # mse, fde = common.test(
    #     path=p,
    #     n_obs=n_obs,
    #     n_pred=n_pred,
    #     batch_size=256,
    #     model=p3vi,
    #     is_cvae=False,
    #     is_m2p3=False
    # )
    # p3vi_mse.append(mse)
    # p3vi_fde.append(fde)
    # print(20 * "#", "\n")

    # print("CI3PP_ATT_BH")
    # ci3pp_ATT_BH = CI3P_ATT_BH(n_predict_frames=n_pred, n_observed_frames=n_obs, embed_dim=256, n_heads=4)
    # ci3pp_ATT_BH.cuda()
    # ci3pp_ATT_BH.load_state_dict(torch.load("./_out/CI3P_ATT_BH/obs60_pred80/embed256_heads4_batch256_lr9.267057602101488e-05/2024-06-18_18-04-42/model_embed256_heads4_batch256_lr9.267057602101488e-05.pth"))
    # mse, fde = common.test(
    #     path=p,
    #     n_obs=60,
    #     n_pred=80,
    #     batch_size=1,
    #     model=ci3pp_ATT_BH,
    #     is_cvae=False,
    #     is_m2p3=False
    # )
    print("CI3PP_ATT_BH")
    ci3pp_ATT_BH = CI3P_ATT_BH(n_predict_frames=n_pred, n_observed_frames=n_obs, embed_dim=128, n_heads=4)
    ci3pp_ATT_BH.cuda()
    ci3pp_ATT_BH.load_state_dict(torch.load("_out/CI3P_ATT_BH/obs60_pred80/embed128_heads4_batch512_lr0.001/2024-07-04_09-23-23/model_embed128_heads4_batch512_lr0.001.pth"))
    mse, fde = common.test(
        path=p,
        n_obs=60,
        n_pred=80,
        batch_size=512,
        model=ci3pp_ATT_BH,
        is_cvae=False,
        is_m2p3=False
    )
    # ci3pp_ATT_BH_mse.append(mse)
    # ci3pp_ATT_BH_fde.append(fde)
    # print(20 * "#", "\n")
    #
    # print("CI3PP_CVAE")
    # ci3pp_cvae = CI3PP_CVAE(predict_frames=n_pred)
    # ci3pp_cvae.cuda()
    # ci3pp_cvae.load_state_dict(torch.load("./_out/CI3PP_CVAE/obs60_pred80/_batch512_lr6.309358225746731e-05/2024-06-18_01-06-31/model__batch512_lr6.309358225746731e-05.pth"))
    # mse, fde = common.test(
    #     path=p,
    #     n_obs=n_obs,
    #     n_pred=n_pred,
    #     batch_size=512,
    #     model=ci3pp_cvae,
    #     is_cvae=True,
    #     is_m2p3=False
    # )
    # ci3pp_CVAE_mse.append(mse)
    # ci3pp_CVAE_fde.append(fde)
    # print(20 * "#", "\n")
    #
    # print("CI3PP_CVAE_ATT")
    # ci3pp_cvae_att = CI3PP_CVAE_ATT(predict_frames=n_pred, n_heads=4, embed_dim=128)
    # ci3pp_cvae_att.cuda()
    # ci3pp_cvae_att.load_state_dict(torch.load("./_out/CI3PP_CVAE_ATT/obs60_pred80/embed128_heads4_batch512_lr6.545381226228296e-05/2024-06-19_04-54-27/model_embed128_heads4_batch512_lr6.545381226228296e-05.pth"))
    # mse, fde = common.test(
    #     path=p,
    #     n_obs=n_obs,
    #     n_pred=n_pred,
    #     batch_size=512,
    #     model=ci3pp_cvae_att,
    #     is_cvae=True,
    #     is_m2p3=False
    # )
    # ci3pp_CVAE_ATT_mse.append(mse)
    # ci3pp_CVAE_ATT_fde.append(fde)
    # print(20 * "#", "\n")



# rows = ["Model"]
# m2p3_row = ["M2P3"]
# p3vi_row = ["P3VI"]
# # ci3pp_ATT_row = ["CI3PP_ATT"]
# # ci3pp_ATT_SH_row = ["CI3PP_ATT_SH"]
# ci3pp_ATT_BH_row = ["CI3PP_ATT_BH"]
# ci3pp_CVAE_row = ["CI3PP_CVAE"]
# ci3pp_CVAE_ATT_row = ["CI3PP_CVAE_ATT"]
#
# #
# for i in range(len(data_paths)):
#     rows.append(f"Scenario {i+1}")
#     m2p3_row.append(f"{m2p3_mse[i]}/{m2p3_fde[i]}")
#     p3vi_row.append(f"{p3vi_mse[i]}/{p3vi_fde[i]}")
#     # ci3pp_ATT_row.append(f"{ci3pp_ATT_mse[i]}/{ci3pp_ATT_fde[i]}")
#     # ci3pp_ATT_SH_row.append(f"{ci3pp_ATT_SH_mse[i]}/{ci3pp_ATT_SH_fde[i]}")
#     ci3pp_ATT_BH_row.append(f"{ci3pp_ATT_BH_mse[i]}/{ci3pp_ATT_BH_fde[i]}")
#     ci3pp_CVAE_row.append(f"{ci3pp_CVAE_mse[i]}/{ci3pp_CVAE_fde[i]}")
#     ci3pp_CVAE_ATT_row.append(f"{ci3pp_CVAE_ATT_mse[i]}/{ci3pp_CVAE_ATT_fde[i]}")
#
# t = PrettyTable(rows)
# t.add_row(m2p3_row)
# t.add_row(p3vi_row)
# # t.add_row(ci3pp_ATT_row)
# # t.add_row(ci3pp_ATT_SH_row)
# t.add_row(ci3pp_ATT_BH_row)
# t.add_row(ci3pp_CVAE_row)
# t.add_row(ci3pp_CVAE_ATT_row)
#
# print(t)
