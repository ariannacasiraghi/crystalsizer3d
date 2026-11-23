import importlib
import os

print('HF_HOME =', os.getenv('HF_HOME'), '| HF_HUB_OFFLINE =', os.getenv('HF_HUB_OFFLINE'))


def check(name):
    try:
        m = importlib.import_module(name)
        print(f'OK   {name} {getattr(m, "__version__", "")}')
    except Exception as e:
        print(f'FAIL {name}: {type(e).__name__}: {e}')


print('--- packages')
for pkg in ['torch', 'torchvision', 'numpy', 'timm', 'transformers', 'kornia', 'mitsuba', 'cv2',
            'geomloss', 'torchjd', 'lightning', 'gigagan_pytorch', 'parti_pytorch', 'skimage',
            'einops', 'omegaconf', 'orjson', 'ruamel.yaml', 'trimesh', 'pymatgen', 'xxhash', 'dotenv']:
    check(pkg)

print('--- mitsuba CUDA variant')
try:
    import mitsuba as mi
    print('variants:', mi.variants())
    mi.set_variant('cuda_ad_rgb')
    print('OK   cuda_ad_rgb')
except Exception as e:
    print(f'FAIL mitsuba cuda: {type(e).__name__}: {e}')

print('--- repo modules')
for mod in ['crystalsizer3d', 'crystalsizer3d.nn.models.timmnet',
            'crystalsizer3d.nn.manager', 'crystalsizer3d.crystal_renderer']:
    check(mod)

print('--- pretrained weights (needs internet unless already cached)')
try:
    import timm
    timm.create_model('regnetz_d8.ra3_in1k', pretrained=True, num_classes=0)
    print('OK   pretrained regnetz_d8.ra3_in1k loaded')
except Exception as e:
    print(f'FAIL pretrained: {type(e).__name__}: {str(e)[:300]}')
