// Add smooth scrolling to all links with the anchor hash (#)

function apply_smooth_transition(){
	document.querySelectorAll('a[href^="#"]').forEach(anchor => {
	  anchor.addEventListener('click', function (e) {
		e.preventDefault();
		
		// Get the target element and its top position
		const targetElement = document.querySelector(this.getAttribute('href'));
		const targetPosition = targetElement.offsetTop;
		
		// Smoothly scroll to the target position over 500ms
		window.scrollTo({
		  top: targetPosition,
		  behavior: 'smooth'
		});
	  });
	});
};

const sideNav = document.querySelector('.side-nav');
const sideNavInv = document.querySelector('.side-nav-inv');
const sideNavContent = document.querySelector('.side-nav-content');
const sideNavInvContent = document.querySelector('.side-nav-inv-content');

// Define the text to show before and after hover
const defaultText = sideNavInvContent.innerHTML;
const hoverText = sideNavContent.innerHTML;
const defaultWidth = '50px';
const defaultHeight = '40px';
const hoverWidth = '180px';
const hoverHeight = '150px';

// Set the default text
sideNavContent.innerHTML = defaultText;
allow_click_trigger = true

function openmenu(){
	apply_smooth_transition();
	setTimeout(() => {
		sideNavContent.innerHTML = hoverText;
		sideNav.style.height = hoverHeight
		sideNav.style.width = hoverWidth
		sideNav.style.opacity = 1.00
		apply_smooth_transition();
	}, 100);
	apply_smooth_transition();
}

function closemenu(){
	apply_smooth_transition();
	setTimeout(() => {
		sideNavContent.innerHTML = defaultText;
		sideNav.style.height = defaultHeight
		sideNav.style.width = defaultWidth
		sideNav.style.opacity = 0.25
		apply_smooth_transition();
	}, 100);
	apply_smooth_transition();
}

sideNav.addEventListener('mouseenter', () => {
	openmenu();
	allow_click_trigger = false;
  });

sideNav.addEventListener('mouseleave', () => {
	closemenu();
	allow_click_trigger = true;
  });

	
if (navigator.userAgent.match(/Android/i)
	|| navigator.userAgent.match(/webOS/i)
	|| navigator.userAgent.match(/iPhone/i)
	|| navigator.userAgent.match(/iPad/i)
	|| navigator.userAgent.match(/iPod/i)
	|| navigator.userAgent.match(/BlackBerry/i)
	|| navigator.userAgent.match(/Windows Phone/i)) {
		sideNav.style.opacity = 0;
}
	

