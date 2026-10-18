import {menu,changeQuantity,total} from './cart.js';
let cart=new Map(), category='전체', orderNumber=100;
const $=selector=>document.querySelector(selector);
const won=value=>value.toLocaleString('ko-KR')+'원';
function renderMenu(){
  $('#categories').innerHTML=['전체','커피','음료','디저트'].map(name=>`<button data-category="${name}" aria-pressed="${category===name}" class="${category===name?'active':''}">${name}</button>`).join('');
  $('#menu').innerHTML=menu.filter(item=>category==='전체'||item.category===category).map(item=>`<button class="menu-card" data-add="${item.id}" aria-label="${item.name} ${won(item.price)} 담기"><span class="food ${item.category==='음료'?'green':item.category==='디저트'?'pink':''}">${item.emoji}</span><span class="item-name">${item.name}</span><span class="description">${item.description}</span><span class="price">${won(item.price)}<span class="add">+</span></span></button>`).join('');
}
function renderCart(){
 const count=[...cart.values()].reduce((a,b)=>a+b,0);
 $('#count').textContent=count; $('#total').textContent=won(total(cart)); $('#pay').disabled=!count; $('#clear').disabled=!count;
 $('#cart').innerHTML=count?[...cart].map(([id,quantity])=>{const item=menu.find(i=>i.id===id);return `<div class="cart-item"><div class="mini-food">${item.emoji}</div><div class="cart-info"><strong>${item.name}</strong><span>${won(item.price)}</span><div class="stepper"><button data-id="${id}" data-delta="-1" aria-label="${item.name} 수량 줄이기">−</button><span>${quantity}</span><button data-id="${id}" data-delta="1" aria-label="${item.name} 수량 늘리기">+</button></div></div><strong>${won(item.price*quantity)}</strong></div>`;}).join(''):'<div class="empty"><span>🛍️</span><strong>아직 담은 메뉴가 없어요</strong><p>원하는 메뉴를 터치해 담아보세요.</p></div>';
}
$('#categories').addEventListener('click',e=>{const button=e.target.closest('[data-category]');if(button){category=button.dataset.category;renderMenu();}});
$('#menu').addEventListener('click',e=>{const button=e.target.closest('[data-add]');if(button){cart=changeQuantity(cart,button.dataset.add,1);renderCart();$('#announcement').textContent=menu.find(i=>i.id===button.dataset.add).name+' 메뉴를 담았습니다.';}});
$('#cart').addEventListener('click',e=>{const button=e.target.closest('[data-delta]');if(button){cart=changeQuantity(cart,button.dataset.id,Number(button.dataset.delta));renderCart();}});
$('#clear').addEventListener('click',()=>{cart=new Map();renderCart();});
$('#pay').addEventListener('click',()=>{
 if(!cart.size)return;
 const service=$('input[name="service"]:checked').value;
 $('#payment-content').innerHTML=`<p class="eyebrow">ONE MORE STEP</p><h2>주문을 확인해주세요</h2><p>${service==='매장'?'매장에서 드시는':'포장'} 주문입니다.</p><ul class="summary">${[...cart].map(([id,q])=>{const i=menu.find(x=>x.id===id);return `<li><span>${i.name} × ${q}</span><strong>${won(i.price*q)}</strong></li>`;}).join('')}</ul><div class="total"><span>결제 금액</span><strong>${won(total(cart))}</strong></div><p class="simulation-note">모의 결제입니다. 카드나 개인정보는 필요하지 않습니다.</p><button class="primary" id="confirm">모의 결제 완료하기</button>`;
 $('#payment').showModal();
 $('#confirm').addEventListener('click',()=>{orderNumber++;cart=new Map();renderCart();$('#payment-content').innerHTML=`<div class="success">✓</div><h2>주문이 완료되었어요!</h2><p>잠시만 기다려주세요. 정성껏 준비할게요.</p><div class="ticket">주문 번호<strong>${orderNumber}</strong><span>${service} · 모의 결제 완료</span></div><button class="primary" id="again">새 주문 시작하기</button>`;$('#again').addEventListener('click',()=>$('#payment').close());$('#again').focus();});
});
renderMenu();renderCart();
