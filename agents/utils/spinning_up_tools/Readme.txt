SpinningUp Utilities
====================

Files in this folder were created by Achiam Joshua in his amazing article Spinning Up in Deep Reinforcement Learning.

Original files can be found on https://github.com/openai/spinningup

Files here were edited for use with purely pytorch

- Used files had removed/commented tensorflow v.1 support so it is no longer required as package
- Moved Buffer class from VPG and PPO to separate class in Utils
- Added Core file for VPG and PPO - named SpinCore
- MLP Actor Critic modified to be used only for discrete and work directly with information passed, instead of trying to access gym spaces
- logx modified to purely output to file with no command line output
- logx modified to have specific file location and not try to access user defaults
