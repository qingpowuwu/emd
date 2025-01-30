window.HELP_IMPROVE_VIDEOJS = false;

// Keep the original Interpolation-related code unchanged
var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;
var interp_images = [];

function preloadInterpolationImages() {
    for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
        var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
        interp_images[i] = new Image();
        interp_images[i].src = path;
    }
}

function setInterpolationImage(i) {
    var image = interp_images[i];
    image.ondragstart = function() { return false; };
    image.oncontextmenu = function() { return false; };
    $('#interpolation-image-wrapper').empty().append(image);
}

//Initialize video comparison function
function initializeVideoComparison() {
    const container = document.querySelector('.video-compare-wrapper');
    const slider = document.querySelector('.compare-slider');
    const firstLayer = document.querySelector('.video-layer:first-child');
    const s3gsVideo = document.getElementById('slider-s3gsVideo');
    const oursVideo = document.getElementById('slider-oursVideo');
    const sidebysideS3gsVideo = document.getElementById('sidebyside-s3gsVideo');
    const sidebysideOursVideo = document.getElementById('sidebyside-oursVideo');
    let isDown = false;

    if (!container || !slider || !firstLayer) return;

    //Improve video synchronization
    function syncVideos() {
        const threshold = 0.01;
        if (Math.abs(s3gsVideo.currentTime - oursVideo.currentTime) > threshold) {
            oursVideo.currentTime = s3gsVideo.currentTime;
        }
    }

    //Add more video event monitoring
    ['play', 'seeking', 'seeked', 'timeupdate'].forEach(event => {
        s3gsVideo.addEventListener(event, syncVideos);
    });

    //Ensure that the video is loaded before playing
    function setupVideo(video) {
        video.addEventListener('loadedmetadata', () => {
            //Set the initial playback speed to 0.3
            video.playbackRate = 0.3;

            const aspectRatio = video.videoWidth / video.videoHeight;
            if (video.parentElement.classList.contains('video-layer')) {
                const wrapper = document.querySelector('.video-compare-wrapper');
                wrapper.style.aspectRatio = `${aspectRatio}`;
            }

            console.log(`Video dimensions: $ {video.videoWidth}x ${video.videoHeight}, Aspect ratio: ${aspectRatio}`);
        });

        video.addEventListener('ended', () => {
            video.currentTime = 0;
            video.play();
        });

        //Set playback speed immediately (without waiting for the loadedmetadata event)
        video.playbackRate = 0.3;
    }

    //Set initial state for all videos
    setupVideo(s3gsVideo);
    setupVideo(oursVideo);
    setupVideo(sidebysideS3gsVideo);
    setupVideo(sidebysideOursVideo);

    //Set initial position (middle)
    slider.style.left = '50%';
    firstLayer.style.clipPath = 'inset(0 50% 0 0)';

    function handleMove(e) {
        if (!isDown) return;
        e.preventDefault();

        const rect = container.getBoundingClientRect();
        const x = e.type.includes('mouse') ? e.pageX : e.touches[0].pageX;
        const leftPosition = Math.max(0, Math.min(x - rect.left, rect.width));
        const percentage = (leftPosition / rect.width) * 100;

        requestAnimationFrame(() => {
            slider.style.left = `${percentage}%`;
            firstLayer.style.clipPath = `inset(0 ${100 - percentage}% 0 0)`;
        });
    }

    // Mouse events
    slider.addEventListener('mousedown', () => isDown = true);
    window.addEventListener('mouseup', () => isDown = false);
    window.addEventListener('mousemove', handleMove);

    // Touch events
    slider.addEventListener('touchstart', () => isDown = true);
    window.addEventListener('touchend', () => isDown = false);
    window.addEventListener('touchmove', handleMove);

    //Add speed control button
    //Add speed control button to the first video group
    const speedControls1 = `
    <div class="speed-controls" style="margin-top: 10px;">
    <button class="button is-small" onclick="changeSliderSpeed(0.1)">0.1x</button>
    <button class="button is-small is-active" onclick="changeSliderSpeed(0.3)">0.3x</button>
    <button class="button is-small" onclick="changeSliderSpeed(0.5)">0.5x</button>
    <button class="button is-small" onclick="changeSliderSpeed(0.8)">0.8x</button>
    <button class="button is-small" onclick="changeSliderSpeed(1.0)">1.0x</button>
    </div>
    `;
    container.insertAdjacentHTML('afterend', speedControls1);

    //Add speed control button to the second video group
    const stackContainer = document.querySelector('.video-stack-container');
    const speedControls2 = `
    <div class="speed-controls" style="margin-top: 10px;">
    <button class="button is-small" onclick="changeStackSpeed(0.1)">0.1x</button>
    <button class="button is-small is-active" onclick="changeStackSpeed(0.3)">0.3x</button>
    <button class="button is-small" onclick="changeStackSpeed(0.5)">0.5x</button>
    <button class="button is-small" onclick="changeStackSpeed(0.8)">0.8x</button>
    <button class="button is-small" onclick="changeStackSpeed(1.0)">1.0x</button>
    </div>
    `;
    stackContainer.insertAdjacentHTML('afterend', speedControls2);

}


//Control the speed of two sets of videos separately
function changeSliderSpeed(speed) {
    const videos = [
        document.getElementById('slider-s3gsVideo'),
        document.getElementById('slider-oursVideo')
    ];

    videos.forEach(video => {
        if (video) video.playbackRate = speed;
    });

    //Update the button style for the sliding comparison video group
    const speedControls = document.querySelector('.video-compare-wrapper + .speed-controls');
    if (speedControls) {
        speedControls.querySelectorAll('.button').forEach(btn => {
            btn.classList.remove('is-active');
            //Convert button text and speed to numbers for comparison
            const btnSpeed = parseFloat(btn.textContent);
            if (btnSpeed === speed) {
                btn.classList.add('is-active');
            }
        });
    }
}

function changeStackSpeed(speed) {
    const videos = [
        document.getElementById('sidebyside-s3gsVideo'),
        document.getElementById('sidebyside-oursVideo')
    ];

    videos.forEach(video => {
        if (video) video.playbackRate = speed;
    });

    //Update button styles for stacked video groups
    const speedControls = document.querySelector('.video-stack-container + .speed-controls');
    if (speedControls) {
        speedControls.querySelectorAll('.button').forEach(btn => {
            btn.classList.remove('is-active');
            //Convert button text and speed to numbers for comparison
            const btnSpeed = parseFloat(btn.textContent);
            if (btnSpeed === speed) {
                btn.classList.add('is-active');
            }
        });
    }
}

//Define video path configuration for two scenarios
const sliderSceneButtons = {
    'scene089': {
        s3gs: './static/videos/self_supervised/S3Gaussian/diff_rgb/089.mp4',
        ours: './static/videos/self_supervised/ours/diff_rgb/089.mp4'
    },
    'scene546': {
        s3gs: './static/videos/self_supervised/S3Gaussian/diff_rgb/546.mp4',
        ours: './static/videos/self_supervised/ours/diff_rgb/546.mp4'
    },
    'scene053': {
        s3gs: './static/videos/self_supervised/S3Gaussian/diff_rgb/053.mp4',
        ours: './static/videos/self_supervised/ours/diff_rgb/053.mp4'
    },
    'scene080': {
        s3gs: './static/videos/self_supervised/S3Gaussian/diff_rgb/080.mp4',
        ours: './static/videos/self_supervised/ours/diff_rgb/080.mp4'
    },
    'scene640': {
        s3gs: './static/videos/self_supervised/S3Gaussian/diff_rgb/640.mp4',
        ours: './static/videos/self_supervised/ours/diff_rgb/640.mp4'
    },
    'scene323': {
        s3gs: './static/videos/self_supervised/S3Gaussian/diff_rgb/323.mp4',
        ours: './static/videos/self_supervised/ours/diff_rgb/323.mp4'
    }
};

const sideBySideSceneButtons = {
    'scene089': {
        s3gs: './static/videos/self_supervised/S3Gaussian/rgb/089.mp4',
        ours: './static/videos/self_supervised/ours/rgb/089.mp4'
    },
    'scene546': {
        s3gs: './static/videos/self_supervised/S3Gaussian/rgb/546.mp4',
        ours: './static/videos/self_supervised/ours/rgb/546.mp4'
    },
    'scene053': {
        s3gs: './static/videos/self_supervised/S3Gaussian/rgb/053.mp4',
        ours: './static/videos/self_supervised/ours/rgb/053.mp4'
    },
    'scene080': {
        s3gs: './static/videos/self_supervised/S3Gaussian/rgb/080.mp4',
        ours: './static/videos/self_supervised/ours/rgb/080.mp4'
    },
    'scene640': {
        s3gs: './static/videos/self_supervised/S3Gaussian/rgb/640.mp4',
        ours: './static/videos/self_supervised/ours/rgb/640.mp4'
    },
    'scene323': {
        s3gs: './static/videos/self_supervised/S3Gaussian/rgb/323.mp4',
        ours: './static/videos/self_supervised/ours/rgb/323.mp4'
    }
};


//Modify video synchronization logic

function syncSideBySideVideos() {
    const s3gsVideo = document.getElementById('sidebyside-s3gsVideo');
    const oursVideo = document.getElementById('sidebyside-oursVideo');

    if (!s3gsVideo || !oursVideo) return;

    const threshold = 0.05; //  The synchronization threshold can be adjusted as needed

    //Active synchronization function
    function synchronize() {
        if (Math.abs(s3gsVideo.currentTime - oursVideo.currentTime) > threshold) {
            //Using requestAnimationFrame to optimize performance
            requestAnimationFrame(() => {
                oursVideo.currentTime = s3gsVideo.currentTime;
            });
        }
    }

    //Regularly check the synchronization status
    let syncInterval;

    function startSync() {
        if (!syncInterval) {
            syncInterval = setInterval(synchronize, 100); //  Check the synchronization status every 100ms
        }
    }
    // clean 
    function stopSync() {
        if (syncInterval) {
            clearInterval(syncInterval);
            syncInterval = null;
        }
    }

    //Video event monitoring
    ['play', 'seeking', 'seeked', 'timeupdate'].forEach(event => {
        s3gsVideo.addEventListener(event, synchronize);
    });

    //Start synchronization check when the video is playing
    s3gsVideo.addEventListener('play', startSync);
    oursVideo.addEventListener('play', startSync);

    //Stop synchronization check when the video pauses
    s3gsVideo.addEventListener('pause', stopSync);
    oursVideo.addEventListener('pause', stopSync);

    //Reset when the video ends
    s3gsVideo.addEventListener('ended', () => {
        stopSync();
        s3gsVideo.currentTime = 0;
        oursVideo.currentTime = 0;
        if (!s3gsVideo.paused) {
            s3gsVideo.play();
            oursVideo.play();
        }
    });

    //Clean up function
    return () => {
        stopSync();
        ['play', 'seeking', 'seeked', 'timeupdate'].forEach(event => {
            s3gsVideo.removeEventListener(event, synchronize);
        });
    };
}


//Video loading function
//Add synchronization initialization in the loadVideos function
function loadVideos(oursUrl, s3gsUrl, oursVideo, s3gsVideo) {
    return new Promise((resolve, reject) => {
        const loadVideo = (video, url) => {
            return new Promise((resolveVideo, rejectVideo) => {
                const currentSpeed = video.playbackRate;
                video.src = url;

                video.onerror = function() {
                    console.error('Video error details:', {
                        error: video.error,
                        errorCode: video.error ? video.error.code : null,
                        errorMessage: video.error ? video.error.message : null,
                        src: video.src
                    });
                    rejectVideo(`Error loading video: ${url}`);
                };

                video.load();
                video.addEventListener('loadeddata', () => {
                    video.playbackRate = currentSpeed;
                    resolveVideo();
                }, { once: true });
            });
        };

        Promise.all([
            loadVideo(s3gsVideo, s3gsUrl),
            loadVideo(oursVideo, oursUrl)
        ]).then(() => {
            //Ensure that the two videos are played synchronously after loading is complete
            oursVideo.currentTime = s3gsVideo.currentTime;

            //If this is a side-by-side video, initialize enhanced synchronization
            if (oursVideo.id === 'sidebyside-oursVideo') {
                syncSideBySideVideos();
            }

            if (!s3gsVideo.paused) {
                oursVideo.play().catch(console.error);
            }
            resolve();
        }).catch(reject);
    });
}


//Initialization button function
function initializeButtons() {
    //Button for handling sliding comparison videos
    const sliderButtons = document.querySelectorAll('[id^="slider-scene"]');
    const sliderOursVideo = document.getElementById('slider-oursVideo');
    const sliderS3gsVideo = document.getElementById('slider-s3gsVideo');

    if (sliderButtons.length && sliderOursVideo && sliderS3gsVideo) {
        sliderButtons.forEach(button => {
            button.addEventListener('click', function() {
                sliderButtons.forEach(btn => btn.classList.remove('is-primary'));
                this.classList.add('is-primary');

                const sceneId = this.id.replace('slider-', '');
                if (sliderSceneButtons[sceneId]) {
                    loadVideos(
                        sliderSceneButtons[sceneId].ours,
                        sliderSceneButtons[sceneId].s3gs,
                        sliderOursVideo,
                        sliderS3gsVideo
                    ).catch(error => {
                        console.error('Error loading slider videos:', error);
                    });
                }
            });
        });
    }

    //Button for processing side-by-side comparison videos
    const sidebysideButtons = document.querySelectorAll('[id^="sidebyside-scene"]');
    const sidebysideOursVideo = document.getElementById('sidebyside-oursVideo');
    const sidebysideS3gsVideo = document.getElementById('sidebyside-s3gsVideo');

    if (sidebysideButtons.length && sidebysideOursVideo && sidebysideS3gsVideo) {
        sidebysideButtons.forEach(button => {
            button.addEventListener('click', function() {
                sidebysideButtons.forEach(btn => btn.classList.remove('is-primary'));
                this.classList.add('is-primary');

                const sceneId = this.id.replace('sidebyside-', '');
                if (sideBySideSceneButtons[sceneId]) {
                    //Pause the current video before loading a new one
                    sidebysideOursVideo.pause();
                    sidebysideS3gsVideo.pause();

                    loadVideos(
                        sideBySideSceneButtons[sceneId].ours,
                        sideBySideSceneButtons[sceneId].s3gs,
                        sidebysideOursVideo,
                        sidebysideS3gsVideo
                    ).then(() => {
                        //Start playing after loading is complete
                        sidebysideS3gsVideo.play().catch(console.error);
                        sidebysideOursVideo.play().catch(console.error);
                    }).catch(error => {
                        console.error('Error loading sidebyside videos:', error);
                    });
                }
            });
        });
    }
}


// StreetGaussian comparison functionality
// Modified version of initializeStreetGaussianComparison
function initializeStreetGaussianComparison() {
    const videoUrls = {
        '026': {
            gt: './static/videos/supervised/ours/gt_rgb/waymo_emb_026.mp4',
            ours: './static/videos/supervised/ours/rgb/waymo_emb_026.mp4',
            diff: './static/videos/supervised/ours/diff_rgb/waymo_emb_026.mp4'
        },
        '108': {
            gt: './static/videos/supervised/ours/gt_rgb/waymo_emb_108.mp4',
            ours: './static/videos/supervised/ours/rgb/waymo_emb_108.mp4',
            diff: './static/videos/supervised/ours/diff_rgb/waymo_emb_108.mp4'
        },
        '150': {
            gt: './static/videos/supervised/ours/gt_rgb/waymo_emb_150.mp4',
            ours: './static/videos/supervised/ours/rgb/waymo_emb_150.mp4',
            diff: './static/videos/supervised/ours/diff_rgb/waymo_emb_150.mp4'
        }
    };

    const videos = {
        gtVideo: document.getElementById('gt-rgb-video'),
        ourVideo: document.getElementById('our-rgb-video'),
        diffVideo: document.getElementById('diff-video')
    };

    // Scene switching functionality
    const buttons = document.querySelectorAll('.scene-buttons .button');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            buttons.forEach(btn => btn.classList.remove('is-info', 'is-selected'));
            this.classList.add('is-info', 'is-selected');

            const scene = this.dataset.scene;
            if (videoUrls[scene]) {
                // Store current playback state
                const wasPlaying = !videos.gtVideo.paused;

                // Update video sources
                videos.gtVideo.src = videoUrls[scene].gt;
                videos.ourVideo.src = videoUrls[scene].ours;
                videos.diffVideo.src = videoUrls[scene].diff;

                // Reload and play all videos
                Object.values(videos).forEach(video => {
                    if (video) {
                        video.load();
                        if (wasPlaying) {
                            video.play().catch(console.error);
                        }
                    }
                });
            }
        });
    });

    // Video synchronization
    function syncVideos(sourceVideo) {
        Object.values(videos).forEach(video => {
            if (video && video !== sourceVideo && Math.abs(video.currentTime - sourceVideo.currentTime) > 0.1) {
                video.currentTime = sourceVideo.currentTime;
            }
        });
    }

    // Add synchronization listeners
    Object.values(videos).forEach(video => {
        if (video) {
            ['play', 'pause', 'seeking', 'seeked', 'timeupdate'].forEach(event => {
                video.addEventListener(event, () => syncVideos(video));
            });

            // Add loop handling
            video.addEventListener('ended', () => {
                video.currentTime = 0;
                video.play().catch(console.error);
            });
        }
    });

    // Initial load
    document.querySelector('.scene-buttons .button[data-scene="026"]').click();
}

// const path for lane change comparison
function generateLaneChangeScenes() {
    const scenes = ['053', '080', '089', '546', '640'];
    const offsets = ['0.5m', '1.0m', '1.5m'];
    const base = './static/videos_2025_01_27-Rebuttals';
    
    return scenes.reduce((acc, scene) => {
        acc[`scene${scene}`] = {
            gt: `${base}/UnchangedGT/ours/${scene}.mp4`,
            ...offsets.reduce((offsetAcc, offset) => {
                offsetAcc[offset] = {
                    s3gs: {
                        left: `${base}/LaneChangeOffset-${offset}/S3Gaussian/left/${scene}.mp4`,
                        right: `${base}/LaneChangeOffset-${offset}/S3Gaussian/right/${scene}.mp4`
                    },
                    ours: {
                        left: `${base}/LaneChangeOffset-${offset}/ours/left/${scene}.mp4`,
                        right: `${base}/LaneChangeOffset-${offset}/ours/right/${scene}.mp4`
                    }
                };
                return offsetAcc;
            }, {})
        };
        return acc;
    }, {});
}

const laneChangeScenes = generateLaneChangeScenes();

// Lane Change Speed Control
function changeLaneVideoSpeed(speed) {
    const videos = [
        document.getElementById('lane-gt-video'),
        document.getElementById('lane-s3gs-video'),
        document.getElementById('lane-ours-video')
    ];

    videos.forEach(video => {
        if (video) {
            video.playbackRate = speed;
        }
    });

    // Update button styles
    const speedControls = document.querySelector('#lane-change-comparison-section .speed-controls');
    if (speedControls) {
        speedControls.querySelectorAll('.button').forEach(btn => {
            btn.classList.remove('is-active');
            const btnSpeed = parseFloat(btn.textContent);
            if (btnSpeed === speed) {
                btn.classList.add('is-active');
            }
        });
    }
}

function initializeLaneChangeComparison() {
    const videos = {
        gtVideo: document.getElementById('lane-gt-video'),
        s3gsVideo: document.getElementById('lane-s3gs-video'),
        oursVideo: document.getElementById('lane-ours-video')
    };

    let currentDirection = 'left';
    let currentOffset = '0.5m';

    // Offset switching functionality
    const offsetButtons = document.querySelectorAll('#lane-offset-buttons .button');
    offsetButtons.forEach(button => {
        button.addEventListener('click', function() {
            offsetButtons.forEach(btn => btn.classList.remove('is-warning', 'is-active'));
            this.classList.add('is-warning', 'is-active');
            
            currentOffset = this.id.replace('lane-offset-', '') + 'm';
            // update offset text
            document.getElementById('current-offset').textContent = currentOffset + ' lateral offset';
            
            // reload videos
            const currentScene = document.querySelector('#lane-change-buttons .is-primary').id.replace('lane-scene', 'scene');
            loadSceneVideos(currentScene, currentDirection, currentOffset);
        });
    });

    // Direction switching functionality
    const directionButtons = document.querySelectorAll('#lane-direction-buttons .button');
    directionButtons.forEach(button => {
        button.addEventListener('click', function() {
            directionButtons.forEach(btn => btn.classList.remove('is-info', 'is-active'));
            this.classList.add('is-info', 'is-active');
            
            currentDirection = this.id.replace('lane-', '');
            const currentScene = document.querySelector('#lane-change-buttons .is-primary').id.replace('lane-scene', 'scene');
            loadSceneVideos(currentScene, currentDirection, currentOffset);
        });
    });

    function loadSceneVideos(scene, direction, offset) {
        if (laneChangeScenes[scene]) {
            const wasPlaying = !videos.gtVideo.paused;
            const currentTime = videos.gtVideo.currentTime;
            const currentSpeed = videos.gtVideo.playbackRate;

            // Update video sources
            videos.gtVideo.src = laneChangeScenes[scene].gt;
            videos.s3gsVideo.src = laneChangeScenes[scene][offset].s3gs[direction];
            videos.oursVideo.src = laneChangeScenes[scene][offset].ours[direction];

            // Reload and restore state
            Object.values(videos).forEach(video => {
                if (video) {
                    video.load();
                    video.currentTime = currentTime;
                    video.playbackRate = currentSpeed;
                    if (wasPlaying) {
                        video.play().catch(console.error);
                    }
                }
            });
        }
    }

    // Scene switching functionality
    const buttons = document.querySelectorAll('#lane-change-buttons .button');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            buttons.forEach(btn => btn.classList.remove('is-primary'));
            this.classList.add('is-primary');

            const scene = 'scene' + this.id.replace('lane-scene', '');
            loadSceneVideos(scene, currentDirection, currentOffset);
        });
    });

    // Video synchronization
    function syncVideos(sourceVideo) {
        Object.values(videos).forEach(video => {
            if (video && video !== sourceVideo && Math.abs(video.currentTime - sourceVideo.currentTime) > 0.1) {
                video.currentTime = sourceVideo.currentTime;
            }
        });
    }

    // Add synchronization listeners
    Object.values(videos).forEach(video => {
        if (video) {
            ['play', 'pause', 'seeking', 'seeked', 'timeupdate'].forEach(event => {
                video.addEventListener(event, () => syncVideos(video));
            });

            video.addEventListener('ended', () => {
                video.currentTime = 0;
                video.play().catch(console.error);
            });
        }
    });

    // Set initial speed
    changeLaneVideoSpeed(0.3);
}

//Initialize all functions
//Modify the initialize All function to add default scene loading
function initializeAll() {

    initializeVideoComparison();
    initializeButtons();
    initializeStreetGaussianComparison();
    initializeLaneChangeComparison(); // Add this line

    // S3Gaussian: default scene trigger
    document.getElementById('slider-scene053').click();
    document.getElementById('sidebyside-scene053').click();
    // Trigger default scene for lane change comparison
    document.getElementById('lane-offset-0.5').click();
    document.getElementById('lane-scene053').click();
    document.getElementById('lane-left').click();
    // StreetGaussian: default scene trigger

    // Preload interpolation images
    preloadInterpolationImages();

    // S3Gaussian: default scene trigger
    $('#interpolation-slider').on('input', function(event) {
        setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    //Other initialization codes
    $(".navbar-burger").click(function() {
        $(".navbar-burger").toggleClass("is-active");
        $(".navbar-menu").toggleClass("is-active");
    });

    // Bulma carousel
    var options = {
        slidesToScroll: 1,
        slidesToShow: 3,
        loop: true,
        infinite: true,
        autoplay: false,
        autoplaySpeed: 3000,
    };

    var carousels = bulmaCarousel.attach('.carousel', options);
    for (var i = 0; i < carousels.length; i++) {
        carousels[i].on('before:show', state => {
            console.log(state);
        });
    }

    bulmaSlider.attach();
}


//Initialize when DOM loading is complete
document.addEventListener('DOMContentLoaded', initializeAll);

// jQuery ready
$(document).ready(function() {
    // Handle video loading errors
    const handleVideoError = (video) => {
        if (video) {
            video.addEventListener('error', function() {
                console.error('Error loading video:', video.src);
            });
        }
    };

    handleVideoError(document.getElementById('slider-oursVideo'));
    handleVideoError(document.getElementById('slider-s3gsVideo'));
    handleVideoError(document.getElementById('sidebyside-oursVideo'));
    handleVideoError(document.getElementById('sidebyside-s3gsVideo'));
});