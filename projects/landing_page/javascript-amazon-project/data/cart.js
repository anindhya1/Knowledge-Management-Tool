export const cart = [];

export function addToCart(matchingItem, productId){

    let cartQuantity = 0;

    if(matchingItem){
        matchingItem.quantity += 1;
    }
    else{
        cart.push({
            productId: productId,
            quantity: 1
        });
    }
    
    cart.forEach((item) =>{
        cartQuantity += item.quantity;
    });

    return cartQuantity;
}