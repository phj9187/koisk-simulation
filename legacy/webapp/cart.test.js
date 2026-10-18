import test from 'node:test';
import assert from 'node:assert/strict';
import {changeQuantity,total} from './cart.js';
test('주문 추가, 수량 변경 및 최종 삭제',()=>{
 let cart=new Map();cart=changeQuantity(cart,'americano',1);cart=changeQuantity(cart,'americano',1);cart=changeQuantity(cart,'cake',1);
 assert.equal(total(cart),12500);
 cart=changeQuantity(cart,'americano',-1);assert.equal(total(cart),9000);
 cart=changeQuantity(cart,'americano',-1);assert.equal(cart.has('americano'),false);
 cart=changeQuantity(cart,'cake',-1);assert.equal(total(cart),0);assert.equal(cart.size,0);
});
test('입력 상태 보존 및 잘못된 메뉴 차단',()=>{const original=new Map();changeQuantity(original,'latte',1);assert.equal(original.size,0);assert.throws(()=>changeQuantity(original,'unknown',1));});
