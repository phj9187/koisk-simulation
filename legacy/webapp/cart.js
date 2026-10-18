export const menu = [
  {id:'americano',name:'아메리카노',description:'깊고 깔끔한 에스프레소의 풍미',price:3500,category:'커피',emoji:'☕'},
  {id:'latte',name:'카페 라떼',description:'고소한 우유와 진한 에스프레소',price:4200,category:'커피',emoji:'🥛'},
  {id:'vanilla',name:'바닐라 라떼',description:'부드럽게 퍼지는 달콤한 바닐라',price:4700,category:'커피',emoji:'☕'},
  {id:'matcha',name:'말차 라떼',description:'제주 말차의 은은하고 깊은 맛',price:4800,category:'음료',emoji:'🍵'},
  {id:'lemon',name:'레몬 에이드',description:'상큼한 레몬과 톡 쏘는 탄산',price:4500,category:'음료',emoji:'🍋'},
  {id:'strawberry',name:'딸기 라떼',description:'달콤한 딸기와 신선한 우유',price:5000,category:'음료',emoji:'🍓'},
  {id:'croissant',name:'버터 크루아상',description:'겹겹이 바삭한 프랑스산 버터',price:3800,category:'디저트',emoji:'🥐'},
  {id:'cake',name:'치즈 케이크',description:'입안에서 녹는 진한 크림치즈',price:5500,category:'디저트',emoji:'🍰'}
];
export function changeQuantity(cart,id,delta){
  if(!menu.some(item=>item.id===id)||!Number.isInteger(delta)) throw new Error('잘못된 메뉴 또는 수량');
  const next = new Map(cart);
  const quantity = Math.max(0,(next.get(id)||0)+delta);
  if(quantity) next.set(id,quantity); else next.delete(id);
  return next;
}
export function total(cart){return [...cart].reduce((sum,[id,quantity])=>sum+menu.find(item=>item.id===id).price*quantity,0);}
