# -*- mode: python -*-

block_cipher = None
data_files = [
         ( './audioanalysis/icons/*.png', 'icons' ),
         ( './audioanalysis/main.ui', '.' ),
         ( 'README.md', '.' ),
         ( 'LICENSE', '.' ),
         ]

a = Analysis(['audioanalysis/__main__.py'],
             pathex=['./audioanalysis', '/Users/new/Documents/Jarvis Lab/jarvis-lab-audio-analysis'],
             binaries=None,
             datas=data_files,
             hiddenimports=['theano.tensor.shared_randomstreams'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='audioanalysis_0.1.0',
          debug=False,
          strip=False,
          upx=True,
          console=True)
          
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='audioanalysis_0.1.0')
