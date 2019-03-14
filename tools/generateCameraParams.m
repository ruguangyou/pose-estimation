clear
load('stereoCalibParameters_360p.mat');

% left camera
matlab2opencv(stereoParams.CameraParameters1.IntrinsicMatrix', 'camLeft', 'parameters.yml', 'w');
matlab2opencv(stereoParams.CameraParameters1.RadialDistortion', 'radDistLeft', 'parameters.yml', 'a');
matlab2opencv(stereoParams.CameraParameters1.TangentialDistortion', 'tanDistLeft', 'parameters.yml', 'a');

% right camera
matlab2opencv(stereoParams.CameraParameters2.IntrinsicMatrix', 'camRight', 'parameters.yml', 'a');
matlab2opencv(stereoParams.CameraParameters2.RadialDistortion', 'radDistRight', 'parameters.yml', 'a');
matlab2opencv(stereoParams.CameraParameters2.TangentialDistortion', 'tanDistRight', 'parameters.yml', 'a');

% extrinsic: left to right
matlab2opencv(stereoParams.RotationOfCamera2, 'rotationLeftToRight', 'parameters.yml', 'a');
matlab2opencv(stereoParams.TranslationOfCamera2', 'translationLeftToRight', 'parameters.yml', 'a');