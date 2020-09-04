const chai = window.chai;
const mocha = window.mocha;
mocha.setup('bdd');

describe('test', () => {
  it('passes', () => {
    chai.expect(1).to.eql(1);
  });
  
  it('fails', () => {
    chai.expect(1).to.eql(2);
  });
});

mocha.run();